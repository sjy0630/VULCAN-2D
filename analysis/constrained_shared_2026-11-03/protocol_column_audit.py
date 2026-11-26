"""Inspect workbook schema/metadata, without fitting or scoring observations."""
from pathlib import Path
from zipfile import ZipFile
import collections
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix('.json')
INVENTORY = ROOT/'analysis/shared_hbn_2026-11-03/data_inventory.json'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def main():
    inventory = json.loads(INVENTORY.read_text())
    result = {'scope': 'schema and metadata inventory only, no model fitting or holdout scoring',
              'books': [], 'limitations': [
                  'Absence in supplied exports does not establish absence from the original instrument records.',
                  'File creation/modification timestamps are not per-observation acquisition times.',
                  'All-sheet schema was inspected; numerical I-V values were not selected or scored.']}
    for book in inventory['books']:
        path = Path(book['recovered_path'])
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != book['sha256']:
            raise ValueError(f'Workbook changed: {path.name}')
        columns = collections.Counter()
        headers = collections.Counter()
        formulas = collections.Counter()
        labels = set()
        with ZipFile(path) as archive:
            names = archive.namelist()
            shared = []
            if 'xl/sharedStrings.xml' in names:
                shared = [''.join(t.text or '' for t in item.iterfind('.//m:t', NS))
                          for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', NS)]
            worksheets = [n for n in names if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n)]
            for name in worksheets:
                sheet = ET.fromstring(archive.read(name))
                for row in sheet.findall('m:sheetData/m:row', NS):
                    head = []
                    for cell in row.findall('m:c', NS):
                        column = re.sub(r'\d', '', cell.attrib['r'])
                        value = cell.find('m:v', NS)
                        inline = ''.join(t.text or '' for t in cell.iterfind('.//m:t', NS))
                        text = value.text if value is not None else inline
                        if text is None or text == '':
                            continue
                        columns[column] += 1
                        formula = cell.find('m:f', NS)
                        if formula is not None:
                            formulas[(column, formula.text)] += 1
                        if cell.get('t') == 's':
                            text = shared[int(text)]
                        if cell.get('t') in ('s', 'inlineStr', 'str'):
                            labels.add(text)
                        if row.get('r') == '1':
                            head.append((column, text))
                    if row.get('r') == '1':
                        headers[tuple(head)] += 1
            metadata_parts = [n for n in names if any(word in n.lower()
                              for word in ('custom', 'connection', 'comment', 'vba'))]
        result['books'].append({'path': str(path.relative_to(ROOT)), 'sha256': digest,
            'worksheet_count': len(worksheets), 'nonempty_cells_by_column': dict(columns),
            'header_variants': [{'header': list(k), 'sheet_count': v} for k, v in headers.items()],
            'formula_variants': [{'column': k[0], 'formula': k[1], 'count': v}
                                 for k, v in formulas.items()],
            'unique_text_labels': sorted(labels), 'extra_metadata_parts': metadata_parts})
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
