# 电荷反馈、慢模态与已有理论的边界

通过Sider Scholar/OpenAlex检索并下载核对：G. Michalek与B. R. Bulka，*Dynamical correlations in electronic transport through a system of coupled quantum dots*，[arXiv:0907.2879](https://arxiv.org/abs/0907.2879)，[所下载PDF](https://arxiv.org/pdf/0907.2879)。检索原始返回存于`charge_dynamics_literature_discovery.json`。它是不同结构的理论参照，不是本项目h-BN器件的机制证据。

已核对摘要、Section II A–C（PDF pp.2–4）、Section III强耦合局部极小值讨论（p.5）及Section IV A（pp.5–6）。论文研究两个大尺寸、连续态密度的电容耦合量子点，各自连接左右电极，讨论电荷涨落、动态库仑阻塞、亚稳极化态及电流噪声。故“相互作用电荷改变电流”“电荷极化出现亚稳态”“不同弛豫时间产生动力学关联”等概念，均不能当成本项目首次提出。

Section II Eq.(11)用连续态密度的正统隧穿率：`Gamma = DeltaF/[e^2 R*(1−exp(−DeltaF/kBT))]`，其中DeltaF取初态自由能减末态自由能。其同一跃迁对的总正反率随能量变化，不满足当前“占据不改变总尝试率”的离散单能级Fermi率假设。因此，本项目的二值精确闭合、旁侧速率从稳态均值消去及慢模态上界，不能直接套到该文模型，也不能把该文的亚稳性借来证明本项目已拟合参数能保持状态。

Section III明确讨论强电容耦合的极化局部极小值，以及通过相邻电荷态进行热激活转移时指数小的速率。这一事实支持继续检查储存态与激活势垒的区分，但并未给出本项目的h-BN缺陷物种、6 nm局域介电环境、零偏保持寿命或SET→RESET交接律。

当前结论：相互作用/电荷阻塞/亚稳态的概念已有文献；新的候选贡献仍须来自目标数据下的定量判据、竞争模型和预测。一次检索及一篇原文不构成完整新颖性核查，本项目两个解析命题的新颖性仍未确认。

PDF与文本保存在上级`verification/2026-09-20/physics-sources/michalek2009-charge-dynamics.pdf/.txt`。文本提取器对一个图形对象报告语法警告，但相关章节及公式文本可读取；另渲染并目视核对了原PDF第3页的Eq.(11)。所有计算结论来自本项目方程和数据，不以文献检索摘要代替验证。
