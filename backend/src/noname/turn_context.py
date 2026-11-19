from __future__ import annotations

import re

from .llm import GeneratedReply

_GAME_NAMES = ("王者荣耀", "王者", "英雄联盟", "和平精英", "原神", "第五人格", "蛋仔派对")
_ACHIEVEMENT_PATTERN = re.compile(r"(排位|上分|段位|赢|胜利|翻盘|MVP|操作|对抗|击杀|压制)", re.IGNORECASE)
_SOCIAL_PATTERN = re.compile(r"(队友|朋友|开黑|一起玩|战队|同学)")
_RELAX_PATTERN = re.compile(r"(放松|解压|休息|缓一缓|轻松|开心|好玩|有意思)")
_SIMPLE_LIKE_PATTERN = re.compile(r"(喜欢|爱玩|很喜欢|就是喜欢|挺喜欢)")
_AUTONOMY_PATTERN = re.compile(r"(别管我|自己决定|不用改|凭什么|控制我|不理解我|没收|管不着)")
_JUDGED_PATTERN = re.compile(r"(为什么大家|都觉得|别人觉得|说我|不理解|当成坏事|游戏不好)")


def _game_name(text: str) -> str | None:
    return next((name for name in _GAME_NAMES if name in text), None)


def _ended_with_question(text: str | None) -> bool:
    if not text:
        return False
    return text.rstrip().endswith(("？", "?"))


def engage_fallback_reply(
    message: str,
    previous_assistant: str | None = None,
) -> GeneratedReply:
    """Ground ENGAGE fallback in what the user explicitly said this turn.

    This is used only when normal generation is unavailable or rejected. It should
    advance from the user's latest information instead of restarting the same
    generic discovery question. Questions are optional: after the assistant has
    just asked one, prefer a reflective statement unless clarification is needed.
    """

    text = message.strip()
    previous_asked = _ended_with_question(previous_assistant)

    if _ACHIEVEMENT_PATTERN.search(text):
        if any(token in text for token in ("排位", "上分", "赢", "胜利", "段位")):
            if previous_asked:
                return GeneratedReply(
                    reply=(
                        "对你来说，排位上分、赢下来本身就很有成就感。"
                        "段位往上走和赢下一局，是你很在意的部分。"
                    ),
                    quick_replies=["赢下来最爽", "操作和对抗", "段位往上升", "都有"],
                )
            return GeneratedReply(
                reply=(
                    "对你来说，排位上分、赢下来本身就很有成就感。"
                    "更吸引你的是赢下来的结果，还是对局里的操作和对抗？"
                ),
                quick_replies=["赢下来最爽", "操作和对抗", "段位往上升", "都有"],
            )
        if previous_asked:
            return GeneratedReply(
                reply=(
                    "对你来说，游戏里的操作和对抗感本身就很有满足感，"
                    "把操作打出来这件事对你很重要。"
                ),
                quick_replies=["操作打出来", "压制对手", "赢关键局", "都有"],
            )
        return GeneratedReply(
            reply=(
                "对你来说，游戏里的操作和对抗感本身就很有满足感。"
                "这种吸引力更偏向把操作打出来，还是压制对手、赢下关键一局？"
            ),
            quick_replies=["操作打出来", "压制对手", "赢关键局", "都有"],
        )

    if _AUTONOMY_PATTERN.search(text):
        if previous_asked:
            return GeneratedReply(
                reply=(
                    "对你来说，自己的时间由自己决定，这一点很重要。"
                    "你并不希望别人直接替你安排或下结论，我理解。"
                ),
                quick_replies=["自己决定", "先听我说", "别急着劝我", "就是这样"],
            )
        return GeneratedReply(
            reply=(
                "对你来说，自己的时间由自己决定，这一点很重要。"
                "你并不希望别人一上来就认定你有问题，对吗？"
            ),
            quick_replies=["自己决定", "先听我说", "别急着劝我", "就是这样"],
        )

    if _JUDGED_PATTERN.search(text):
        if previous_asked:
            return GeneratedReply(
                reply=(
                    "对你来说，玩游戏本身不一定就是坏事；"
                    "你不太希望别人一上来就把游戏当成问题。"
                ),
                quick_replies=["先了解再说", "不是所有游戏都坏", "我有分寸", "都有"],
            )
        return GeneratedReply(
            reply=(
                "对你来说，玩游戏本身不一定就是坏事；"
                "你不太希望别人一上来就把游戏当成问题。"
                "你更希望他们先理解哪一部分？"
            ),
            quick_replies=["先了解再说", "不是所有游戏都坏", "我有分寸", "都有"],
        )

    if _SOCIAL_PATTERN.search(text):
        if previous_asked:
            return GeneratedReply(
                reply=(
                    "对你来说，和队友、朋友一起玩是很重要的部分，"
                    "配合和固定有人一起玩的感觉都在里面。"
                ),
                quick_replies=["一起配合", "有人陪着玩", "固定开黑", "都有"],
            )
        return GeneratedReply(
            reply=(
                "对你来说，和队友、朋友一起玩是很重要的部分。"
                "你更喜欢一起配合的感觉，还是有人固定陪你玩的感觉？"
            ),
            quick_replies=["一起配合", "有人陪着玩", "固定开黑", "都有"],
        )

    if _RELAX_PATTERN.search(text):
        if previous_asked:
            return GeneratedReply(
                reply=(
                    "对你来说，游戏确实能带来放松和好玩的感觉，"
                    "这就是它现在很直接的吸引力。"
                ),
                quick_replies=["赢的时候", "和朋友一起", "操作顺的时候", "随便玩都行"],
            )
        return GeneratedReply(
            reply=(
                "对你来说，游戏确实能带来放松和好玩的感觉。"
                "通常什么样的一局最容易让你觉得真的放松下来？"
            ),
            quick_replies=["赢的时候", "和朋友一起", "操作顺的时候", "随便玩都行"],
        )

    game = _game_name(text)
    if game:
        if previous_asked:
            return GeneratedReply(
                reply=(
                    f"对你来说，{game}确实有你喜欢的地方，"
                    "最近玩的乐趣已经说得比较清楚了。"
                ),
                quick_replies=["排位赢了", "操作很顺", "和朋友开黑", "说不上来"],
            )
        return GeneratedReply(
            reply=f"对你来说，{game}确实有你喜欢的地方。最近什么样的一局最容易让你觉得特别好玩？",
            quick_replies=["排位赢了", "操作很顺", "和朋友开黑", "说不上来"],
        )

    if _SIMPLE_LIKE_PATTERN.search(text):
        if previous_asked:
            return GeneratedReply(
                reply=(
                    "对你来说，你就是很享受玩游戏这件事本身，"
                    "这种喜欢已经表达得很清楚了。"
                ),
                quick_replies=["排位上分", "操作和对抗", "和朋友一起", "其他"],
            )
        if previous_assistant and any(
            marker in previous_assistant
            for marker in ("最吸引", "特别的意义", "哪些方面", "什么地方")
        ):
            return GeneratedReply(
                reply=(
                    "对你来说，你现在更想表达的是：你就是很享受玩游戏这件事本身。"
                    "最近有没有哪一局让你特别开心或者印象很深？"
                ),
                quick_replies=["赢了一局", "和朋友一起", "打出好操作", "想不起来"],
            )
        return GeneratedReply(
            reply="对你来说，你确实很喜欢玩游戏。最近哪一种游戏体验最让你觉得有意思？",
            quick_replies=["排位上分", "操作和对抗", "和朋友一起", "其他"],
        )

    if previous_asked:
        return GeneratedReply(
            reply=(
                "对你来说，你刚刚补充的这部分我先接住，"
                "不急着继续追问。"
            ),
            quick_replies=["游戏本身", "最近的体验", "和朋友一起", "先随便聊聊"],
        )

    return GeneratedReply(
        reply="对你来说，这件事里最值得先聊清楚的是哪一部分？",
        quick_replies=["游戏本身", "最近的体验", "和朋友一起", "先随便聊聊"],
    )
