// electron-builder afterPack hook — ad-hoc code-sign the macOS app.
//
// Apple Silicon refuses to run unsigned/broken-signature arm64 binaries
// (immediate SIGKILL), so even without a paid Developer ID we must ad-hoc sign
// ("-"). The app MUST be built in a non-iCloud-synced directory (see
// scripts/build_mac.sh) — otherwise the iCloud file provider keeps re-adding
// FinderInfo/provenance xattrs and codesign fails with "resource fork, Finder
// information, or similar detritus not allowed". In a clean (e.g. /tmp)
// location, a single xattr clear + a normal deep sign works.
const { execSync } = require('child_process')
const path = require('path')

exports.default = async function afterPack(context) {
  if (context.electronPlatformName !== 'darwin') return
  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`)
  console.log('[afterPack] ad-hoc signing:', app)
  execSync(`xattr -cr "${app}"`, { stdio: 'inherit' })
  execSync(`codesign --force --deep --sign - "${app}"`, { stdio: 'inherit' })
  execSync(`codesign --verify --deep --strict "${app}"`, { stdio: 'inherit' })
  console.log('[afterPack] ad-hoc signature applied + verified')
}
