# LaTeX 实验报告使用指南

## 📄 文档概览

该LaTeX文档（`LATEX_REPORT.tex`）包含了完整的LVI-SAM视觉里程计改进实验报告，包括：

- ✅ 完整的章节结构（绪论、方法论、结果、结论等）
- ✅ 所有定量与定性分析
- ✅ 表格引用（表格宽度调整为48%）
- ✅ 图像引用（对比图表、轨迹图等）
- ✅ 代码示例与伪代码
- ✅ 数学公式与推导
- ✅ 参考文献

## 🚀 快速开始

### 1. 检查依赖

#### Linux (Ubuntu/Debian)

```bash
# 安装 TeX Live (完整版本)
sudo apt-get install texlive-full

# 或者最小化安装
sudo apt-get install texlive-xetex texlive-latex-extra texlive-lang-chinese
```

#### macOS

```bash
# 使用 Homebrew
brew cask install mactex

# 或使用 MacTeX 官方安装器
# 访问: https://tug.org/mactex/
```

#### Windows

```
访问: https://miktex.org/download
选择 MiKTeX 并安装（包含 xelatex）
```

### 2. 编译文档

#### 自动编译脚本

```bash
chmod +x compile_latex.sh
./compile_latex.sh
```

#### 手动编译

```bash
# 编译三次以确保目录和交叉引用正确
xelatex -interaction=nonstopmode LATEX_REPORT.tex
xelatex -interaction=nonstopmode LATEX_REPORT.tex
xelatex -interaction=nonstopmode LATEX_REPORT.tex

# 清理临时文件
rm -f *.aux *.log *.out *.toc *.lof *.lot
```

### 3. 查看PDF

```bash
# Linux
evince LATEX_REPORT.pdf

# macOS
open LATEX_REPORT.pdf

# Windows
start LATEX_REPORT.pdf
```

## 📊 文档结构

```
┌─────────────────────────────────────┐
│      第一章: 绪论                  │
│  • 研究背景 • 研究意义 • 主要贡献 │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第二章: 相关工作               │
│  • LVI-SAM系统 • 特征跟踪方法     │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第三章: 改进方法论             │
│  • 多描述符提取 • 自适应分布      │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第四章: 实验设置               │
│  • 数据集 • 参数配置              │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第五章: 实验结果与分析         │
│  • 定量结果 • 定性分析 • 可视化   │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第六章: 实现细节               │
│  • Python实现 • C++接口           │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第七章: 性能分析               │
│  • 复杂度分析 • 可扩展性          │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第八章: 后续改进规划           │
│  • Phase 4-6 计划                 │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      第九章: 结论与展望             │
│  • 主要结论 • 工作展望            │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│      附录A: 文件清单               │
│  • 代码、数据、文档               │
└─────────────────────────────────────┘
```

## 🎨 文档特点

### 1. 表格设置

所有表格宽度均设为 **0.48\textwidth**，确保在一页内显示完整：

```latex
\resizebox{0.48\textwidth}{!}{
  \begin{tabular}{...}
    ...
  \end{tabular}
}
```

**包含的表格：**
- 表1: 数据集信息
- 表2: 特征跟踪参数
- 表3: APE详细对比
- 表4: ATE和ARE对比
- 表5: 计算复杂度
- 表6: 可扩展性分析
- 表7: 传感器参数
- 和更多...

### 2. 图像引用

所有图像使用 `\includegraphics` 命令进行引用：

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\textwidth]{evaluation_results/comparison.png}
  \caption{Baseline vs Enhanced VO 性能对比柱状图}
  \label{fig:comparison}
\end{figure}
```

**包含的图像：**
- 图1: 性能对比柱状图 (evaluation_results/comparison.png)
- 图2: 基线轨迹分析 (evaluation_results/baseline_trajectory.png)
- 图3: 改进轨迹分析 (evaluation_results/enhanced_vo_trajectory.png)

### 3. 中文支持

使用 **XeLaTeX** 和 **ctex** 包支持完整的中文排版：

```latex
\documentclass[12pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\usepackage{xeCJK}
\setCJKmainfont{SimSun}
```

### 4. 数学公式

包含高质量的数学排版：

$$\text{APE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\mathbf{p}_i^{est} - \mathbf{p}_i^{ref}\|_2^2}$$

### 5. 代码块

使用 `listings` 包进行代码高亮：

```python
class EnhancedVisualOdometry:
    def track_image(self, img, timestamp):
        """处理新图像进行特征跟踪"""
        ...
```

## 📝 自定义文档

### 修改标题和作者

在文档开头修改以下部分：

```latex
\title{\textbf{\Large 基于增强视觉特征跟踪的LVI-SAM系统优化\\
        Enhanced Visual Odometry Improvement for LVI-SAM}}
\author{您的名字}
\date{\today}
```

### 添加新章节

```latex
\section{新章节标题}

\subsection{小节标题}

正文内容...
```

### 添加表格

```latex
\begin{table}[H]
\centering
\resizebox{0.48\textwidth}{!}{
\begin{tabular}{|l|c|c|}
\hline
\rowcolor{rowcolor}
\textbf{列1} & \textbf{列2} & \textbf{列3} \\
\hline
数据1 & 数据2 & 数据3 \\
\hline
\end{tabular}
}
\caption{表格标题}
\label{tab:标签}
\end{table}
```

### 引用表格和图像

```latex
如表 \ref{tab:标签} 所示...
如图 \ref{fig:标签} 所示...
```

## 🔧 常见问题

### Q1: 编译失败，提示找不到图像文件

**解决方案：**
- 确保图像文件存在于指定位置
- 检查路径是否正确（相对于 LATEX_REPORT.tex）
- 图像文件应该在 `evaluation_results/` 目录下

```
evaluation_results/
├── comparison.png              # ✓ 对比图表
├── baseline_trajectory.png     # ✓ 基线轨迹图
└── enhanced_vo_trajectory.png  # ✓ 改进轨迹图
```

### Q2: 中文显示为乱码

**解决方案：**
- 确保使用 `xelatex` 而不是 `pdflatex`
- 确保系统安装了中文字体（如 SimSun）
- 使用 `fc-list` 检查可用字体

```bash
fc-list :lang=zh
```

### Q3: PDF文件过大

**解决方案：**
- 使用 `gs` 命令压缩PDF

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
   -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile=LATEX_REPORT_compressed.pdf \
   LATEX_REPORT.pdf
```

### Q4: 页码和目录不正确

**解决方案：**
- 多次编译文档（通常需要3次）
- 第一次生成辅助文件
- 第二次生成目录
- 第三次确保交叉引用正确

## 📚 输出示例

编译成功后，您将获得包含以下内容的PDF：

```
┌────────────────────────────────────┐
│  标题页                            │
│  • 文章标题、作者、日期            │
└────────────────────────────────────┘
            ↓
┌────────────────────────────────────┐
│  目录                              │
│  • 各章节列表                      │
│  • 页码自动生成                    │
└────────────────────────────────────┘
            ↓
┌────────────────────────────────────┐
│  正文 (9章 + 附录)                 │
│  • 所有章节内容                    │
│  • 表格、图像、公式                │
│  • 代码示例                        │
└────────────────────────────────────┘
            ↓
┌────────────────────────────────────┐
│  参考文献                          │
│  • 5篇关键论文                     │
└────────────────────────────────────┘
            ↓
┌────────────────────────────────────┐
│  页码: 第 X 页，共 Y 页            │
│  • 页脚自动添加                    │
└────────────────────────────────────┘
```

## 📖 LaTeX语法快速参考

### 基础结构

```latex
\documentclass{article}        % 文档类型
\usepackage{包名}              % 导入包

\begin{document}
  \section{一级标题}           % 章
  \subsection{二级标题}        % 节
  \subsubsection{三级标题}     % 小节
\end{document}
```

### 文本格式

```latex
\textbf{加粗}                  % 加粗
\textit{斜体}                  % 斜体
\underline{下划线}             % 下划线
{\color{red} 红色文本}         % 彩色文本
```

### 列表

```latex
\begin{itemize}
  \item 项目1
  \item 项目2
\end{itemize}

\begin{enumerate}
  \item 第一项
  \item 第二项
\end{enumerate}
```

### 数学模式

```latex
$x^2 + y^2 = z^2$            % 行内公式
\[ x^2 + y^2 = z^2 \]        % 行间公式
\begin{equation}
  x^2 + y^2 = z^2
\label{eq:pythagoras}
\end{equation}
```

## 🎓 学习资源

- **官方文档**: https://www.latex-project.org/
- **Overleaf教程**: https://www.overleaf.com/learn
- **中文LaTeX指南**: https://www.ctan.org/tex-archive/info/lshort/chinese

## ✅ 检查清单

编译前请确认：

- [ ] `LATEX_REPORT.tex` 文件存在
- [ ] `evaluation_results/` 目录和图像文件存在
- [ ] 系统已安装 `xelatex`
- [ ] 系统已安装中文字体（如SimSun）
- [ ] 有足够的磁盘空间

编译后请验证：

- [ ] 生成了 `LATEX_REPORT.pdf` 文件
- [ ] PDF能正常打开
- [ ] 所有中文正常显示
- [ ] 所有图表都被正确引用
- [ ] 页码和目录正确

## 📞 技术支持

如遇到问题，请检查：

1. 编译器版本
2. 系统字体配置
3. 文件路径和权限
4. 依赖包是否齐全

## 🎉 完成！

当您看到 "✅ 所有步骤完成!" 时，说明编译成功了。

您现在有了一份专业的实验报告PDF文档，可以用于：
- 学术论文发表
- 研究成果展示
- 学位论文撰写
- 项目报告文档

---

**最后更新**: 2026-01-17  
**格式**: XeLaTeX + ctex (中文支持)  
**页数**: 20-25页  
**文件大小**: 约 5-8 MB
