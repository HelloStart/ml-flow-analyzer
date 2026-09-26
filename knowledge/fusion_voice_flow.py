from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_PATH = Path(__file__).with_name("fusion_voice_flow.png")

for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(font_name.lower() in font.name.lower() for font in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [font_name]
        break
rcParams["axes.unicode_minus"] = False


def draw_box(axis, x, y, width, height, title, detail, color):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor=color, edgecolor="#36566f", linewidth=1.4,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height * 0.66, title, ha="center", va="center",
              fontsize=11, fontweight="bold", color="#17344d")
    axis.text(x + width / 2, y + height * 0.31, detail, ha="center", va="center",
              fontsize=8.5, color="#29485e", wrap=True)


def arrow(axis, start, end, label):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                   linewidth=1.5, color="#d4654b"))
    axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.045, label,
              ha="center", va="bottom", fontsize=8.5, color="#a84935")


def main():
    figure, axis = plt.subplots(figsize=(11, 5.8), dpi=140)
    figure.patch.set_facecolor("#fffdf8")
    axis.set_facecolor("#fffdf8")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.02, 0.95, "端云协同语音助手：ML + GenAI 融合流程", fontsize=16,
              fontweight="bold", color="#17344d")
    axis.text(0.02, 0.895, "本地持续感知，唤醒后再调用云端语音理解与生成服务", fontsize=10, color="#496476")

    axis.text(0.23, 0.81, "端侧：ESP32-S3", ha="center", fontsize=11, fontweight="bold", color="#2d6f70")
    axis.text(0.76, 0.81, "云端：语音与生成服务", ha="center", fontsize=11, fontweight="bold", color="#6e5a9d")
    axis.plot([0.5, 0.5], [0.11, 0.84], color="#cfc9bc", linewidth=1.2, linestyle="--")

    draw_box(axis, 0.05, 0.57, 0.18, 0.15, "麦克风音频", "16 kHz 音频流\n持续输入", "#dceef0")
    draw_box(axis, 0.29, 0.57, 0.18, 0.15, "ESP-SR + WakeNet", "音频前端处理\n唤醒词二分类", "#dceef0")
    draw_box(axis, 0.55, 0.57, 0.17, 0.15, "ASR", "语音转文本", "#ece3f5")
    draw_box(axis, 0.78, 0.57, 0.17, 0.15, "LLM", "理解上下文\n生成回答", "#ece3f5")
    draw_box(axis, 0.66, 0.25, 0.17, 0.15, "TTS", "回答转 PCM 音频", "#ece3f5")
    draw_box(axis, 0.24, 0.25, 0.19, 0.15, "I2S 功放 + 扬声器", "播放“在”或\n完整语音回答", "#dceef0")

    arrow(axis, (0.23, 0.645), (0.29, 0.645), "本地处理")
    arrow(axis, (0.47, 0.645), (0.55, 0.645), "唤醒后上传语音")
    arrow(axis, (0.72, 0.645), (0.78, 0.645), "文本")
    arrow(axis, (0.86, 0.57), (0.79, 0.40), "生成回答")
    arrow(axis, (0.66, 0.325), (0.43, 0.325), "回传 PCM 音频")

    axis.text(0.38, 0.47, "只在触发后跨端传输", ha="center", fontsize=9,
              fontweight="bold", color="#a84935")
    axis.text(0.24, 0.09, "低延迟、低功耗、持续运行", ha="center", fontsize=9, color="#2d6f70")
    axis.text(0.76, 0.09, "较大模型、理解与生成能力", ha="center", fontsize=9, color="#6e5a9d")
    figure.tight_layout(pad=1)
    figure.savefig(OUTPUT_PATH, facecolor=figure.get_facecolor())


if __name__ == "__main__":
    main()