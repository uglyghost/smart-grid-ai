import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu,
  Eye,
  Map,
  Activity,
  Zap,
  Database,
  Server,
  Navigation,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  Users,
  Radio,
  Terminal,
  Brain,
  ScanEye
} from 'lucide-react';

// --- Config & Data ---

const PROJECT_INFO = {
  title: "变电站空间智能与具身巡检自导航技术",
  subtitle: "面向复杂复合环境的自主移动机器人(AMR)空间感知与决策系统",
  course: "课程4：变电站空间智能与具身巡检自导航技术研究",
  partners: ["北京邮电大学"]
};

const EXPECTED_INDICATORS = [
  { label: "目标识别准确率", value: "≥95%", icon: <Eye className="w-6 h-6 text-blue-400" /> },
  { label: "语义分类精度", value: "≥85%", icon: <Users className="w-6 h-6 text-green-400" /> },
  { label: "语音识别率", value: "≥97%", icon: <Activity className="w-6 h-6 text-purple-400" /> },
  { label: "动态导航延迟", value: "≤10s", icon: <Zap className="w-6 h-6 text-yellow-400" /> },
  { label: "定位偏差", value: "≤cm级", icon: <Map className="w-6 h-6 text-red-400" /> },
  { label: "障碍物识别率", value: "≥98%", icon: <ShieldCheck className="w-6 h-6 text-indigo-400" /> },
];

const DATASETS = [
  { name: "Robotics Transformer X", url: "https://robotics-transformer-x.github.io/" },
  { name: "EmbodiedVerse", url: "https://huggingface.co/spaces/BAAI/EmbodiedVerse" },
  { name: "GPT4Scene & VLN-R1", url: "https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1" }
];

// --- Formula Components ---

const FORMULAS = {
  // 1. GSPO 目标函数
  gspo_objective: {
    title: "GSPO 目标函数 (Objective)",
    MathComponent: () => (
      <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" className="text-lg">
        <mrow>
          <msub><mi>J</mi><mtext>GSPO</mtext></msub>
          <mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo>
          <mo>=</mo>
          <msub><mi mathvariant="double-struck">E</mi>
            <mrow>
              <mi>x</mi><mo>,</mo>
              <mo stretchy="false">{'{'}</mo><msub><mi>y</mi><mi>i</mi></msub><msubsup><mo stretchy="false">{'}'}</mo><mn>1</mn><mi>G</mi></msubsup>
            </mrow>
          </msub>
          <mfenced open="[" close="]">
            <mrow>
              <mfrac><mn>1</mn><mi>G</mi></mfrac>
              <munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover>
              <mi>min</mi>
              <mfenced open="(" close=")">
                <mrow>
                  <msub><mi>r</mi><mi>t</mi></msub><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo>
                  <msub><mover><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub>
                  <mo>,</mo>
                  <mtext>clip</mtext>
                  <mo stretchy="false">(</mo>
                  <msub><mi>r</mi><mi>t</mi></msub><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo>
                  <mo>,</mo>
                  <mn>1</mn><mo>−</mo><mi>ε</mi>
                  <mo>,</mo>
                  <mn>1</mn><mo>+</mo><mi>ε</mi>
                  <mo stretchy="false">)</mo>
                  <msub><mover><mi>A</mi><mo>^</mo></mover><mi>i</mi></msub>
                </mrow>
              </mfenced>
            </mrow>
          </mfenced>
        </mrow>
      </math>
    ),
    parts: [
      { symbol: "G", meaning: "序列组 (Group)", detail: "GSPO 采样的序列集合大小，通过整组序列的评估来优化长程决策。" },
      { symbol: "r_t(θ)", meaning: "重要性比率", detail: "新旧策略概率之比，用于在 Off-policy 更新中修正偏差。" }
    ]
  },
  // 2. Control Variates 梯度估计
  control_variates: {
    title: "Control Variates 梯度估计",
    MathComponent: () => (
      <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" className="text-lg">
        <mrow>
          <msub><mo>∇</mo><mi>θ</mi></msub>
          <mi>J</mi>
          <mo>≈</mo>
          <mfrac><mn>1</mn><mi>G</mi></mfrac>
          <munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>G</mi></munderover>
          <mfenced open="[" close="]">
            <mrow>
              <mo stretchy="false">(</mo>
              <mi>R</mi><mo stretchy="false">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo stretchy="false">)</mo>
              <mo>−</mo>
              <mi mathcolor="#4ade80">b</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo>
              <mo stretchy="false">)</mo>
              <msub><mo>∇</mo><mi>θ</mi></msub>
              <mi>log</mi>
              <msub><mi>π</mi><mi>θ</mi></msub>
              <mo stretchy="false">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>|</mo><mi>x</mi><mo stretchy="false">)</mo>
            </mrow>
          </mfenced>
        </mrow>
      </math>
    ),
    parts: [
      { symbol: "b(x)", meaning: "控制变量 (Control Variates)", detail: "基线函数（通常为组内奖励均值）。它与动作无关，在不引入偏差的前提下显著降低梯度方差。" },
      { symbol: "R(y_i) - b(x)", meaning: "优势 (Advantage)", detail: "当前序列相对于基线的表现，决定了策略更新的方向和幅度。" }
    ]
  },
  // 3. 多模态对比学习
  contrastive_alignment: {
    title: "多模态对比学习 (Image-Text Alignment)",
    MathComponent: () => (
      <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" className="text-lg">
        <mrow>
          <msub><mi mathvariant="script">L</mi><mtext>CL</mtext></msub>
          <mo>=</mo>
          <mo>−</mo>
          <munder><mo>∑</mo><mi>i</mi></munder>
          <mi>log</mi>
          <mfrac>
            <mrow>
              <mi>exp</mi>
              <mo stretchy="false">(</mo>
              <mtext>sim</mtext>
              <mo stretchy="false">(</mo><msub><mi>V</mi><mi>i</mi></msub><mo>,</mo><msub><mi>T</mi><mi>i</mi></msub><mo stretchy="false">)</mo>
              <mo>/</mo><mi>τ</mi>
              <mo stretchy="false">)</mo>
            </mrow>
            <mrow>
              <munder><mo>∑</mo><mi>j</mi></munder>
              <mi>exp</mi>
              <mo stretchy="false">(</mo>
              <mtext>sim</mtext>
              <mo stretchy="false">(</mo><msub><mi>V</mi><mi>i</mi></msub><mo>,</mo><msub><mi>T</mi><mi>j</mi></msub><mo stretchy="false">)</mo>
              <mo>/</mo><mi>τ</mi>
              <mo stretchy="false">)</mo>
            </mrow>
          </mfrac>
        </mrow>
      </math>
    ),
    parts: [
      { symbol: "sim(V, T)", meaning: "特征对齐", detail: "计算视觉特征 V (变电站环境图像) 与文本特征 T (巡检指令) 的余弦相似度。" },
      { symbol: "Contrastive", meaning: "对比机制", detail: "拉近匹配的图文对（正样本），推开不匹配的图文对（负样本），赋予机器人理解语义的能力。" }
    ]
  }
};

// --- Sub-Components ---

const Navbar = ({ scrollTo }) => (
  <nav className="fixed top-0 left-0 w-full z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-700 text-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex items-center justify-between h-16">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => scrollTo('hero')}>
          <Zap className="w-8 h-8 text-blue-500" />
          <span className="font-bold text-xl tracking-wider">SmartGrid<span className="text-blue-500">AI</span></span>
        </div>
        <div className="hidden md:block">
          <div className="ml-10 flex items-baseline space-x-8">
            {['核心架构', '理论算法', '模拟演示', '相关资源'].map((item, idx) => {
              const ids = ['architecture', 'algorithms', 'simulation', 'resources'];
              return (
                <button key={item} onClick={() => scrollTo(ids[idx])} className="hover:text-blue-400 transition-colors px-3 py-2 rounded-md text-sm font-medium">
                  {item}
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  </nav>
);

const Hero = () => (
  <section id="hero" className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 bg-slate-950 overflow-hidden">
    <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>
    <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 mb-8">
        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
        课程4：变电站空间智能专项研究
      </div>
      <h1 className="text-5xl md:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-200 via-white to-blue-200 mb-6 tracking-tight leading-tight">
        具身智能 · <span className="text-blue-500">空间感知</span> · 自主导航
      </h1>
      <p className="mt-4 max-w-2xl mx-auto text-xl text-slate-400">
        建立变电站复合环境下的空间智能模型，实现“感知-决策-行为”的持续优化。
        基于 VLN 与 NMIT 机制，打造下一代电力巡检具身智能体。
      </p>
      <div className="mt-16 pt-8 border-t border-slate-800/50 flex flex-wrap justify-center gap-8 opacity-70">
        {PROJECT_INFO.partners.map(p => (
          <span key={p} className="text-slate-300 font-semibold text-lg">{p}</span>
        ))}
      </div>
    </div>
  </section>
);

const Architecture = () => {
  const [showIndicators, setShowIndicators] = useState(false);

  return (
    <section id="architecture" className="py-20 bg-slate-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">云边端协同 · 智能架构</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            通过边缘智能节点与中心云端的实时交互，实现从数据采集到智能决策的全链路闭环。
          </p>
        </div>

        {/* Nodes Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 relative mb-20">
          <div className="hidden lg:block absolute top-12 left-[20%] right-[20%] h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 z-0"></div>

          {/* Terminal */}
          <div className="relative z-10 bg-slate-800 border border-slate-700 p-8 rounded-2xl hover:border-blue-500/50 transition-all group">
            <div className="w-16 h-16 bg-blue-900/50 rounded-xl flex items-center justify-center mb-6">
              <Radio className="w-8 h-8 text-blue-400" />
            </div>
            <h3 className="text-xl font-bold mb-3">端侧：多模态感知</h3>
            <ul className="text-slate-400 space-y-2 text-sm">
              <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>激光雷达/视觉/红外采集</li>
              <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>实时SLAM定位</li>
            </ul>
          </div>

          {/* Edge */}
          <div className="relative z-10 bg-slate-800 border border-slate-700 p-8 rounded-2xl hover:border-purple-500/50 transition-all group shadow-2xl shadow-purple-900/20">
            <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-purple-600 text-white px-4 py-1 rounded-full text-xs font-bold tracking-wider">核心处理</div>
            <div className="w-16 h-16 bg-purple-900/50 rounded-xl flex items-center justify-center mb-6">
              <Cpu className="w-8 h-8 text-purple-400" />
            </div>
            <h3 className="text-xl font-bold mb-3">边侧：空间智能模型</h3>
            <ul className="text-slate-400 space-y-2 text-sm">
              <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>空间语义定义与分析</li>
              <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>VLN 视觉语言导航规划</li>
            </ul>
          </div>

          {/* Cloud */}
          <div className="relative z-10 bg-slate-800 border border-slate-700 p-8 rounded-2xl hover:border-blue-500/50 transition-all group">
            <div className="w-16 h-16 bg-indigo-900/50 rounded-xl flex items-center justify-center mb-6">
              <Server className="w-8 h-8 text-indigo-400" />
            </div>
            <h3 className="text-xl font-bold mb-3">云端：知识与训练</h3>
            <ul className="text-slate-400 space-y-2 text-sm">
              <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-indigo-500 rounded-full"></div>多智能体强化学习训练</li>
              <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-indigo-500 rounded-full"></div>全局知识图谱更新</li>
            </ul>
          </div>
        </div>

        {/* Toggleable KPIs */}
        <div className="flex flex-col items-center">
          <button
            onClick={() => setShowIndicators(!showIndicators)}
            className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors text-sm font-semibold bg-slate-800/50 px-6 py-3 rounded-full border border-blue-500/20 hover:border-blue-500/50"
          >
            {showIndicators ? '收起预期指标' : '点击查看预期指标'}
            {showIndicators ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          <div className={`mt-8 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 w-full transition-all duration-500 overflow-hidden ${showIndicators ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}`}>
            {EXPECTED_INDICATORS.map((kpi, idx) => (
              <div key={idx} className="bg-slate-800/50 border border-slate-700/50 p-4 rounded-xl text-center">
                <div className="flex justify-center mb-2">{kpi.icon}</div>
                <div className="text-2xl font-bold text-white mb-1">{kpi.value}</div>
                <div className="text-xs text-slate-400">{kpi.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

// Math Card Component
const TheoryCard = ({ formulaKey, theoryTitle, theoryDesc, theoryTags }) => {
  const formula = FORMULAS[formulaKey];
  const MathContent = formula.MathComponent;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-slate-900 rounded-2xl border border-slate-800 p-6 shadow-lg hover:border-slate-700 transition-colors">

      {/* Left: Theory (5 Columns) */}
      <div className="lg:col-span-5 flex flex-col justify-center border-b lg:border-b-0 lg:border-r border-slate-800 pb-6 lg:pb-0 lg:pr-6">
        <h3 className="text-xl font-bold text-white mb-3 flex items-center gap-2">
          <Brain className="w-5 h-5 text-blue-500" />
          {theoryTitle}
        </h3>
        <p className="text-slate-400 text-sm leading-relaxed mb-4 text-justify">
          {theoryDesc}
        </p>
        <div className="flex flex-wrap gap-2 mt-auto">
          {theoryTags.map((tag, i) => (
            <span key={i} className="text-xs px-2 py-1 bg-slate-800 text-slate-300 rounded border border-slate-700">
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Right: Math (7 Columns) */}
      <div className="lg:col-span-7 flex flex-col">
        <div className="bg-slate-950 rounded-lg p-5 border border-slate-800 flex items-center justify-center min-h-[100px] overflow-x-auto">
          <MathContent />
        </div>

        {/* Legend */}
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2">
          {formula.parts.map((part, idx) => (
            <div key={idx} className="flex gap-2 items-start text-xs">
              <span className="font-mono text-blue-400 bg-blue-900/20 px-1 rounded flex-shrink-0">{part.symbol}</span>
              <span className="text-slate-500">
                <strong className="text-slate-300">{part.meaning}：</strong>
                {part.detail}
              </span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};

const Algorithms = () => (
  <section id="algorithms" className="py-20 bg-slate-950">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
          核心算法：<span className="text-blue-500">GSPO 与 多模态对齐</span>
        </h2>
        <p className="text-slate-400 max-w-2xl mx-auto">
          融合组序列策略优化 (GSPO)、控制变量技术与多模态对比学习，
          构建高鲁棒性的具身智能导航系统。
        </p>
      </div>

      <div className="space-y-8">
        {/* Card 1: GSPO Overview */}
        <TheoryCard
          formulaKey="gspo_objective"
          theoryTitle="GSPO 组序列策略优化"
          theoryDesc="GSPO 引入了“组 (Group)”的概念来解决长程导航问题。它每次采样一组完整的导航路径，并基于整组的表现来更新策略。这种方法比传统的单步更新更能捕捉长期依赖关系，确保机器人在复杂的变电站环境中不会迷失方向。"
          theoryTags={["Group Sampling", "Long-term Planning", "RL"]}
        />

        {/* Card 2: Control Variates */}
        <TheoryCard
          formulaKey="control_variates"
          theoryTitle="Control Variates (方差缩减)"
          theoryDesc="策略梯度算法通常面临高方差的问题，导致训练不稳定。我们在 GSPO 中引入 Control Variates 技术，利用“组内平均奖励”作为基线 (Baseline)。这在数学上保证了梯度估计的无偏性 (Unbiased)，同时显著降低了方差，加快了模型的收敛速度。"
          theoryTags={["Variance Reduction", "Baseline", "Optimization"]}
        />

        {/* Card 3: Contrastive Learning */}
        <TheoryCard
          formulaKey="contrastive_alignment"
          theoryTitle="多模态对比学习 (特征对齐)"
          theoryDesc="为了让机器人“听懂”指令并“看懂”环境，我们采用对比学习来对齐视觉特征与文本特征。通过最大化匹配图文对（正样本）的相似度，最小化不匹配对（负样本）的相似度，模型能够在特征空间中将指令意图与视觉感知紧密关联。"
          theoryTags={["Image-Text Alignment", "CLIP", "Representation Learning"]}
        />
      </div>
    </div>
  </section>
);

const SimulationView = () => {
  const [logs, setLogs] = useState([]);
  const canvasRef = useRef(null);

  // Multi-Agent Simulation State
  const simState = useRef({
    agents: [
      { id: 'A01', pos: { x: 50, y: 350 }, target: { x: 750, y: 50 }, velocity: { x: 0, y: 0 }, path: [], mode: 'GO', color: '#3b82f6', speed: 1.2 },
      { id: 'A02', pos: { x: 100, y: 380 }, target: { x: 600, y: 80 }, velocity: { x: 0, y: 0 }, path: [], mode: 'GO', color: '#60a5fa', speed: 1.0 },
      { id: 'A03', pos: { x: 30, y: 300 }, target: { x: 700, y: 150 }, velocity: { x: 0, y: 0 }, path: [], mode: 'GO', color: '#93c5fd', speed: 1.4 }
    ],
    obstacles: [
      { x: 400, y: 200, r: 25 },
      { x: 250, y: 120, r: 30 },
      { x: 600, y: 280, r: 28 }
    ],
    lastLogTime: 0
  });

  const addLog = (msg) => {
    setLogs(prev => {
      if (prev.length > 0 && prev[0] === msg) return prev;
      return [msg, ...prev].slice(0, 8);
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationId;
    let angle = 0;

    const draw = () => {
      // 1. Physics & Logic Update
      const state = simState.current;
      const { agents, obstacles } = state;

      agents.forEach(agent => {
        // Calculate vectors
        const dx = agent.target.x - agent.pos.x;
        const dy = agent.target.y - agent.pos.y;
        const distToTarget = Math.sqrt(dx*dx + dy*dy);

        // Normalized direction to target (Attractive Force)
        let vx = (dx / distToTarget) * agent.speed;
        let vy = (dy / distToTarget) * agent.speed;

        // Obstacle Avoidance (Repulsive Force)
        let avoiding = false;
        obstacles.forEach(obs => {
          const odx = agent.pos.x - obs.x;
          const ody = agent.pos.y - obs.y;
          const dist = Math.sqrt(odx*odx + ody*ody);
          const safeDist = 100; // Increased detection range for smoother avoidance

          if (dist < safeDist) {
            avoiding = true;
            const force = (safeDist - dist) / safeDist;
            // Stronger repulsion
            vx += (odx / dist) * force * 2.0;
            vy += (ody / dist) * force * 2.0;
          }
        });

        // Update Position
        agent.pos.x += vx;
        agent.pos.y += vy;

        // Record Path (every 10 frames to save memory and make it smoother)
        if (Math.random() > 0.8) {
          agent.path.push({ ...agent.pos });
          if (agent.path.length > 200) agent.path.shift();
        }

        // Check Target Reached
        if (distToTarget < 15) {
          if (agent.mode === 'GO') {
            agent.mode = 'RETURN';
            // Set return target close to start but slightly random
            agent.target = { x: 50 + Math.random()*50, y: 300 + Math.random()*50 };
            addLog(`[${agent.id}] 到达指定点，开始返航采集`);
            agent.path = []; // Optional: Clear path
          } else {
            agent.mode = 'GO';
            // Set new go target
            agent.target = { x: 700 + Math.random()*50, y: 50 + Math.random()*50 };
            addLog(`[${agent.id}] 返回基站，数据同步完成`);
            agent.path = [];
          }
        } else if (avoiding && Math.random() > 0.98) {
           // Low probability log to avoid spamming
           addLog(`[${agent.id}] 触发避障预警`);
        }
      });

      // 2. Rendering
      ctx.fillStyle = '#0f172a'; // Background
      ctx.fillRect(0, 0, 800, 400);

      // Grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for(let i=0; i<800; i+=40) {
        ctx.beginPath(); ctx.moveTo(i,0); ctx.lineTo(i,400); ctx.stroke();
      }
      for(let i=0; i<400; i+=40) {
        ctx.beginPath(); ctx.moveTo(0,i); ctx.lineTo(800,i); ctx.stroke();
      }

      // Draw Obstacles
      obstacles.forEach(obs => {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.15)'; // Red glow
        ctx.beginPath(); ctx.arc(obs.x, obs.y, obs.r + 15, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = '#ef4444'; // Red core
        ctx.beginPath(); ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI*2); ctx.fill();
      });

      // Draw Agents
      agents.forEach(agent => {
        // Draw Path
        if (agent.path.length > 1) {
          ctx.strokeStyle = agent.color; // Use agent color
          ctx.globalAlpha = 0.4;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(agent.path[0].x, agent.path[0].y);
          for(let i=1; i<agent.path.length; i++) {
            ctx.lineTo(agent.path[i].x, agent.path[i].y);
          }
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }

        // Draw Target
        ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        ctx.beginPath(); ctx.arc(agent.target.x, agent.target.y, 10, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = '#10b981';
        ctx.beginPath(); ctx.arc(agent.target.x, agent.target.y, 4, 0, Math.PI*2); ctx.fill();

        // Draw Robot Body
        ctx.shadowBlur = 10;
        ctx.shadowColor = agent.color;
        ctx.fillStyle = agent.color;
        ctx.beginPath(); ctx.arc(agent.pos.x, agent.pos.y, 6, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;

        // Agent Label
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText(agent.id, agent.pos.x + 10, agent.pos.y - 10);
      });

      angle += 0.05;
      animationId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationId);
  }, []);

  return (
    <section id="simulation" className="py-20 bg-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-12 text-center">具身智能体 <span className="text-blue-500">模拟终端</span></h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 bg-slate-950 p-4 rounded-3xl border border-slate-800 shadow-2xl">
          <div className="lg:col-span-2 relative h-[400px] bg-black rounded-2xl overflow-hidden border border-slate-800">
            <canvas ref={canvasRef} width={800} height={400} className="w-full h-full object-cover" />
            <div className="absolute top-4 left-4 z-10 flex gap-2">
               <span className="bg-red-500/20 text-red-500 px-2 py-1 rounded text-xs font-mono animate-pulse flex items-center gap-1">
                 <div className="w-2 h-2 rounded-full bg-red-500"></div> LIVE
               </span>
               <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded text-xs font-mono">
                 MULTI-AGENT SLAM ACTIVE
               </span>
            </div>
          </div>
          <div className="flex flex-col gap-4">
            <div className="flex-1 bg-slate-900 rounded-2xl border border-slate-800 p-4 font-mono text-xs overflow-hidden flex flex-col">
              <div className="flex items-center gap-2 text-slate-400 mb-3 pb-2 border-b border-slate-800"><Terminal className="w-4 h-4" /> 实时日志 (Real-time Logs)</div>
              <div className="space-y-2 overflow-y-auto flex-1">
                {logs.map((log, idx) => (
                  <div key={idx} className={`border-l-2 pl-2 transition-all ${idx === 0 ? 'border-blue-500 text-blue-300 font-bold' : 'border-slate-700 text-slate-500'}`}>
                    <span className="opacity-50 mr-2">[{new Date().toLocaleTimeString().split(' ')[0]}]</span>
                    {log}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

const Footer = () => (
  <footer id="resources" className="bg-slate-950 pt-20 pb-10 border-t border-slate-900">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-16">
        <div>
          <h3 className="text-2xl font-bold text-white mb-6">相关资源 & 数据集</h3>
          <ul className="space-y-4">
            {DATASETS.map((d, i) => (
              <li key={i} className="flex items-center gap-3"><Database className="w-5 h-5 text-blue-400" /><a href={d.url} target="_blank" className="text-slate-400 hover:text-white">{d.name}</a></li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="text-2xl font-bold text-white mb-6">项目团队</h3>
          <div className="bg-slate-900 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center gap-2 mb-2"><Users className="text-blue-500" /><span className="text-white font-bold">核心成员</span></div>
            <p className="text-slate-400 text-sm">许长桥, 关建峰, 王目, 杨树杰, 陈星延, 周赞, 肖寒</p>
          </div>
        </div>
      </div>
      <div className="border-t border-slate-900 pt-8 text-center text-slate-600 text-sm"><p>© 2024 Course 4 Project Team.</p></div>
    </div>
  </footer>
);

export default function App() {
  const scrollTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  return (
    <div className="bg-slate-950 min-h-screen text-slate-200 font-sans">
      <Navbar scrollTo={scrollTo} />
      <main><Hero /><Architecture /><Algorithms /><SimulationView /></main>
      <Footer />
    </div>
  );
}