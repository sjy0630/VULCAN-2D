# 保存节点上的共同乘积根单调性补充检查

在实际lambda搜索执行前，仅使用前阶段lambda0保存积分节点的支持、均值、
A/B/y，检查H=ln(alpha beta/kappa r)在RESET正系数区间的导数上界。
不求值新lambda、不执行参数根、电路、反演或ODE，不改变既定31点方案。

P表示SET4外扫两窗，N表示RESET4外扫两窗。固定四配置全部保留。
先检查R_P0>S_P>T_P且两个y_P为负、S_N<T_N且y_N均正、两组局部电压
支持分离。均值用保存的lambda0正积分权重计算。

在lambda>=lambda0的RESET正系数域S_N<R_N<T_N，有

H'=-mu_NL-R_N'[1/(T_N-R_N)+2/(R_N-S_N)]
   +mu_PL-R_P'/(R_P-T_P)+2R_P'/(R_P-S_P)。

正指数和的mu导数为加权方差，故mu_NH(lambda)>=mu_NH0。
利用R_N>=S_N及mu_NL<=v_NL,max，可得

Cminus=S_N*(mu_NH0-v_NL,max)*(1+sqrt(2))^2/(T_N-S_N)。

同时R_P增加，R_P/(R_P-S_P)递减，故

Cplus=v_PL,max+2*R_P0/(R_P0-S_P)*(v_PH,max-v_PL,min)。

若Cminus>Cplus，则H'<=Cplus-Cminus<0。结合边界H→+∞/-∞，该有限
节点问题恰有一个乘积关系根。仍需b域、其余窗口和前向验证；不得推广到
未知连续实验轨迹。若条件失败，保留失败，仍沿用有限搜索局限。

实际计算仅四配置的支持/加权均值和标量不等式。新lambda点评估、参数根、
模型调用均为零。输入哈希和全部中间量应落盘，接受独立推导复核。
