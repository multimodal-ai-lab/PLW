from __future__ import annotations
import copy
import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Union
from PIL import Image
from plw.utils.prompt_dataset import PromptDataset


class Role(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


RESET = "\x1b[0m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
MAGENTA = "\x1b[35m"
ROLE_COLORS = {
    Role.SYSTEM: MAGENTA,
    Role.ASSISTANT: "\x1b[44;1m",
    Role.USER: "\x1b[48;2;208;78;16m",
}


@dataclass
class Message:
    content: Union[str, Image.Image]
    role: Role = Role.USER
    is_generated: bool = False
    position: int = None

    def is_assistant(self) -> bool:
        return self.role == Role.ASSISTANT

    def is_text(self) -> bool:
        return isinstance(self.content, str)

    def is_image(self) -> bool:
        return isinstance(self.content, Image.Image)

    def is_image_ref(self) -> bool:
        """True when content is an unresolved image reference (after JSON load)."""
        return (
            isinstance(self.content, dict) and self.content.get("type") == "image_ref"
        )

    def to_dict(self, img_rel_path: Optional[str] = None) -> dict:
        """
        Serialise to a JSON-safe dict.
        For image messages, ``img_rel_path`` must be supplied
        (e.g. ``'my_dataset_images/0_2.png'``).
        """
        if self.is_image():
            if img_rel_path is None:
                raise ValueError(
                    "img_rel_path is required when serialising an image message."
                )
            content_field = {"type": "image_ref", "path": img_rel_path}
        else:
            content_field = self.content
        return {
            "role": self.role.value,
            "content": content_field,
            "is_generated": self.is_generated,
        }

    @classmethod
    def from_dict(cls, d: dict, position: int = None) -> Message:
        """
        Deserialise from dict.  Image content stays as an ``image_ref`` dict
        here; actual pixel data is loaded later by
        ``ChatHistory.load_images()``.
        """
        return cls(
            content=d["content"],
            role=Role(d["role"]),
            is_generated=d.get("is_generated", False),
            position=position,
        )

    def resolve_image(self, base_dir: Path) -> None:
        """Replace the image_ref dict with the actual PIL Image loaded from disk."""
        if not self.is_image_ref():
            return
        img_path = base_dir / self.content["path"]
        self.content = Image.open(img_path).copy()

    def __str__(self) -> str:
        color = ROLE_COLORS[self.role]
        label = self.role.value.upper() + (
            "" if not self.is_generated else " (generated)"
        )
        header = f"{color} {label} {RESET}"
        if self.is_image():
            w, h = self.content.size
            mode = self.content.mode
            body = f"{CYAN}🖼  Image  {w}×{h}  [{mode}]{RESET}"
        elif self.is_image_ref():
            body = f"{CYAN}🖼  Image ref → {self.content['path']}{RESET}"
        else:
            body = self.content
        return f"{header}\n  {body}"


import re
from pathlib import Path
from typing import List, Union

MAGENTA = "\x1b[35m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
_KW_TERM_BG = "\x1b[48;2;180;35;35m"
_KW_TERM_FG = "\x1b[38;2;255;255;255m"
_KW_TERM_END = "\x1b[0m"
_KW_IMG_BG = (200, 60, 60)
_KW_IMG_FG = (255, 255, 255)
_KW_PAD_H = 4
_KW_PAD_V = 1
_KW_RADIUS = 4
_KW_GAP = 3


class ChatHistory:

    def __init__(self, messages, system_prompt: str = None):
        self.system_prompt = system_prompt
        self.messages = messages
        self.keywords: List[str] = []

    def add_keywords(self, *keywords: str) -> None:
        """Add one or more highlight keywords to this chat history."""
        self.keywords.extend((k.strip() for k in keywords if k.strip()))

    def _segments(self, text: str) -> list[tuple[str, bool]]:
        """
        Split *text* into (fragment, is_keyword) pairs.
        Matched fragments are case-insensitive.
        """
        if not self.keywords:
            return [(text, False)]
        pattern = re.compile(
            "(" + "|".join((re.escape(k) for k in self.keywords)) + ")", re.IGNORECASE
        )
        kw_lower = {k.lower() for k in self.keywords}
        return [
            (part, part.lower() in kw_lower) for part in pattern.split(text) if part
        ]

    def _highlight_text_ansi(self, text: str) -> str:
        """Return *text* with ANSI-highlighted keywords (terminal output)."""
        if not self.keywords:
            return text
        pattern = re.compile(
            "(" + "|".join((re.escape(k) for k in self.keywords)) + ")", re.IGNORECASE
        )
        return pattern.sub(
            lambda m: f"{_KW_TERM_BG}{_KW_TERM_FG}{m.group()}{_KW_TERM_END}", text
        )

    def _draw_highlighted_line(
        self, draw, x0: int, y: int, line: str, font, lh: int, text_col: tuple
    ) -> None:
        """
        Render one wrapped line onto *draw* starting at (x0, y).

        Pill geometry for keyword fragments:
          - _KW_GAP of space is added before and after each pill
          - pill rect: rx0 = x  ..  rx1 = x + fw + 2*_KW_PAD_H
          - text drawn at x + _KW_PAD_H  →  equal left AND right inset
        """
        x = x0
        for fragment, is_kw in self._segments(line):
            fw = int(font.getlength(fragment))
            if is_kw:
                x += _KW_GAP
                rx0 = x
                ry0 = y - _KW_PAD_V
                rx1 = x + fw + 2 * _KW_PAD_H
                ry1 = y + lh - 2 + _KW_PAD_V
                draw.rounded_rectangle(
                    [rx0, ry0, rx1, ry1], radius=_KW_RADIUS, fill=_KW_IMG_BG
                )
                draw.text((x + _KW_PAD_H, y), fragment, font=font, fill=_KW_IMG_FG)
                x += fw + 2 * _KW_PAD_H + _KW_GAP
            else:
                draw.text((x, y), fragment, font=font, fill=text_col)
                x += fw

    @classmethod
    def from_user(cls, *contents, system_prompt: str = "") -> "ChatHistory":
        """Shorthand when every element is a user turn."""
        return cls(
            messages=[
                Message(content=c, role=Role.USER, position=pos)
                for pos, c in enumerate(contents)
            ],
            system_prompt=system_prompt,
        )

    def to_dict(self, chat_idx: int, img_dir: Path) -> dict:
        serialised_messages = []
        for msg in self.messages:
            if msg.is_image():
                filename = f"{chat_idx}_{msg.position}.png"
                img_dir.mkdir(parents=True, exist_ok=True)
                msg.content.save(img_dir / filename)
                rel = Path(img_dir.name) / filename
                serialised_messages.append(msg.to_dict(img_rel_path=str(rel)))
            else:
                serialised_messages.append(msg.to_dict())
        return {"messages": serialised_messages}

    @classmethod
    def from_dict(cls, d: dict) -> "ChatHistory":
        return cls(
            messages=[
                Message.from_dict(m, position=pos)
                for pos, m in enumerate(d["messages"])
            ]
        )

    def load_images(self, base_dir: Path) -> None:
        for msg in self.messages:
            if msg.is_image_ref():
                msg.resolve_image(base_dir)

    def append(self, content, role=None, is_generated: bool = False) -> None:
        role = role or Role.USER
        self.messages.append(
            Message(
                content=content,
                role=role,
                is_generated=is_generated,
                position=len(self),
            )
        )

    def prepend(self, content, role=None, is_generated: bool = False) -> None:
        role = role or Role.USER
        self.messages.insert(
            0,
            Message(
                content=content,
                role=role,
                is_generated=is_generated,
                position=len(self),
            ),
        )

    def __add__(self, other):
        if not isinstance(other, ChatHistory):
            return NotImplemented
        combined_messages = self.messages + other.messages
        system_prompt = self.system_prompt or other.system_prompt
        combined_keywords = self.keywords + other.keywords
        combined_chat = ChatHistory(
            messages=combined_messages, system_prompt=system_prompt
        )
        combined_chat.add_keywords(*combined_keywords)
        return combined_chat

    def is_singleton(self) -> bool:
        return len(self.messages) == 1

    def contains_images(self) -> bool:
        return any((m.is_image() for m in self.messages))

    def contains_image_refs(self) -> bool:
        return any((m.is_image_ref() for m in self.messages))

    @property
    def user_messages(self):
        return [m for m in self.messages if m.role == Role.USER]

    @property
    def system_messages(self):
        return [m for m in self.messages if m.role == Role.SYSTEM]

    @property
    def texts(self) -> List[str]:
        return [m.content for m in self.messages if m.is_text()]

    @property
    def images(self):
        return [m.content for m in self.messages if m.is_image()]

    def __str__(self) -> str:
        width = 60
        header = "ChatHistory"
        if self.system_prompt:
            header += f"  ·  {self.system_prompt}"
        msg_lines = [self._highlight_text_ansi(str(msg)) for msg in self.messages]
        lines = [
            f"{MAGENTA}{'─' * width}",
            f" {header}",
            f"{'─' * width}{RESET}",
            *msg_lines,
            f"{DIM}{'─' * width}{RESET}",
        ]
        return "\n".join(lines)

    def pretty_print(self) -> None:
        print(self)

    def to_image(self, canvas_width: int = 1024):
        """Render the chat history as a messenger-style image (WhatsApp-like)."""
        from PIL import Image, ImageDraw, ImageFont, ImageFilter

        BG = (245, 245, 247)
        BUBBLE_USER = (255, 230, 215)
        BUBBLE_OTHER = (235, 240, 252)
        BUBBLE_IMG = (248, 248, 248)
        TEXT_COL = (30, 30, 30)
        LABEL_USER = (180, 60, 5)
        LABEL_OTHER = (55, 90, 190)
        BORDER_USER = (230, 170, 130)
        BORDER_OTHER = (190, 205, 235)
        HEADER_BG = (255, 255, 255)
        HEADER_FG = (30, 30, 30)
        HEADER_LINE = (220, 220, 225)
        AVATAR_BG = (208, 78, 16)
        TAIL = 9
        RADIUS = 16
        H_PAD = 14
        V_PAD = 10
        MARGIN = 16
        BUBBLE_MAX_W = int(canvas_width * 0.7)
        SPACING = 6
        GROUP_GAP = 14
        HEADER_H = 64

        def _font(bold: bool = False, size: int = 15) -> ImageFont.FreeTypeFont:
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf",
                "/usr/share/fonts/liberation/LiberationSans{}-Regular.ttf",
                "/usr/share/fonts/gnu-free/FreeSans{}.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial{}.ttf",
                "C:/Windows/Fonts/arial{}.ttf",
            ]
            suffix = "-Bold" if bold else ""
            for pattern in candidates:
                path = pattern.format(suffix)
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue
            return ImageFont.load_default()

        f_body = _font(size=15)
        f_bold = _font(bold=True, size=15)
        f_label = _font(bold=True, size=12)
        f_hdr = _font(bold=True, size=17)
        f_sub = _font(size=12)
        lh_body = f_body.getbbox("Ag")[3] + 4

        def wrap(text: str, max_px: int) -> list[str]:
            words = text.split()
            lines, cur = ([], "")
            for w in words:
                candidate = (cur + " " + w).strip()
                if f_body.getlength(candidate) <= max_px:
                    cur = candidate
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            return lines or [""]

        def _line_width(text: str) -> int:
            w = 0
            for fragment, is_kw in self._segments(text):
                w += int(f_body.getlength(fragment))
                if is_kw:
                    w += 2 * _KW_PAD_H + 2 * _KW_GAP
            return w

        records = []
        y = HEADER_H + SPACING * 5
        prev_role = None
        for msg in self.messages:
            is_user = msg.role.name == "USER"
            role_changed = msg.role != prev_role
            if role_changed and prev_role is not None:
                y += GROUP_GAP
            inner_w = BUBBLE_MAX_W - 2 * H_PAD
            if msg.is_image():
                img = msg.content
                scale = min(1.0, inner_w / img.width)
                iw, ih = (int(img.width * scale), int(img.height * scale))
                bw = iw + 2 * H_PAD
                bh = ih + 2 * V_PAD
                lines_ = None
                thumb_ = msg.content.resize((iw, ih), Image.LANCZOS)
            else:
                lines_ = wrap(msg.content, inner_w)
                max_line_px = max((_line_width(l) for l in lines_))
                bw = min(BUBBLE_MAX_W, max_line_px + 2 * H_PAD + 2)
                bh = len(lines_) * lh_body + 2 * V_PAD
                thumb_ = None
            bx = canvas_width - MARGIN - bw - TAIL if is_user else MARGIN + TAIL
            records.append(
                dict(
                    msg=msg,
                    is_user=is_user,
                    role_changed=role_changed,
                    x=bx,
                    y=y,
                    w=bw,
                    h=bh,
                    lines=lines_,
                    thumb=thumb_,
                )
            )
            y += bh + SPACING
            prev_role = msg.role
        total_h = y + SPACING * 3
        canvas = Image.new("RGB", (canvas_width, total_h), BG)
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle([0, 0, canvas_width, HEADER_H], fill=HEADER_BG)
        draw.line([0, HEADER_H, canvas_width, HEADER_H], fill=HEADER_LINE, width=1)
        av_cx, av_cy, av_r = (38, HEADER_H // 2, 22)
        draw.ellipse(
            [av_cx - av_r, av_cy - av_r, av_cx + av_r, av_cy + av_r], fill=AVATAR_BG
        )
        draw.text((av_cx, av_cy), "AI", font=f_bold, fill=(255, 255, 255), anchor="mm")
        tx = av_cx + av_r + 12
        draw.text((tx, av_cy - 11), "ChatHistory", font=f_hdr, fill=HEADER_FG)
        draw.text(
            (tx, av_cy + 5),
            f"{len(self.messages)} messages",
            font=f_sub,
            fill=(150, 150, 160),
        )
        for rec in records:
            x, y, w, h = (rec["x"], rec["y"], rec["w"], rec["h"])
            is_user = rec["is_user"]
            is_img_msg = rec["thumb"] is not None
            color = (
                BUBBLE_IMG if is_img_msg else BUBBLE_USER if is_user else BUBBLE_OTHER
            )
            border_col = (
                (BORDER_USER if is_user else BORDER_OTHER)
                if not is_img_msg
                else (220, 220, 220)
            )
            shadow_img = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
            sd = ImageDraw.Draw(shadow_img)
            sd.rounded_rectangle(
                [4, 4, w + 4, h + 4], radius=RADIUS, fill=(0, 0, 0, 28)
            )
            shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(3))
            canvas.paste(shadow_img, (x - 4, y - 2), shadow_img)
            if is_user:
                tail_pts = [
                    (x + w - 2, y + h - RADIUS + 2),
                    (x + w + TAIL, y + h - 6),
                    (x + w - 2, y + h - 6),
                ]
            else:
                tail_pts = [
                    (x + 2, y + h - RADIUS + 2),
                    (x - TAIL, y + h - 6),
                    (x + 2, y + h - 6),
                ]
            draw.polygon(tail_pts, fill=color)
            draw.rounded_rectangle(
                [x, y, x + w, y + h],
                radius=RADIUS,
                fill=color,
                outline=border_col,
                width=1,
            )
            if rec["role_changed"] or rec is records[0]:
                label = "User" if is_user else "Assistant"
                label_col = LABEL_USER if is_user else LABEL_OTHER
                label_y = y - 18
                label_x = (
                    x + H_PAD
                    if not is_user
                    else x + w - int(f_label.getlength(label)) - H_PAD
                )
                draw.text((label_x, label_y), label, font=f_label, fill=label_col)
            if rec["thumb"] is not None:
                canvas.paste(rec["thumb"], (x + H_PAD, y + V_PAD))
            else:
                ty = y + V_PAD
                for line in rec["lines"]:
                    self._draw_highlighted_line(
                        draw, x + H_PAD, ty, line, f_body, lh_body, TEXT_COL
                    )
                    ty += lh_body
        return canvas

    def __iter__(self):
        return (msg.content for msg in self.messages)

    def __getitem__(self, idx):
        return self.messages[idx]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        sys = f"system_prompt={self.system_prompt!r}, " if self.system_prompt else ""
        kw = f"keywords={self.keywords!r}, " if self.keywords else ""
        return f"ChatHistory({sys}{kw}messages={self.messages!r})"


class ChatHistoryDataset:

    def __init__(
        self,
        name: str,
        chat_histories: List[ChatHistory],
        num_images_per_chat: int = 1,
        seed: int = None,
    ):
        self.name = name
        self.chat_histories = chat_histories
        self.num_images_per_chat = num_images_per_chat
        self.seed = seed

    @classmethod
    def from_prompt_dataset(cls, prompt_dataset: PromptDataset) -> ChatHistoryDataset:
        return ChatHistoryDataset(
            name=prompt_dataset.name,
            chat_histories=[ChatHistory.from_user(p) for p in prompt_dataset],
            num_images_per_chat=prompt_dataset.num_images_per_prompt,
            seed=prompt_dataset.seed,
        )

    def save_to_jsonl(self, jsonl_path: Union[str, Path]) -> None:
        """
        Persist the dataset to a .jsonl file (one ``ChatHistory`` per line).

        Images are saved to a sibling directory named ``{stem}_images/``.
        Paths stored inside the JSONL are relative to ``jsonl_path``'s parent,
        so the whole dataset can be moved without breaking anything.

        On-disk layout::

            my_dataset.jsonl
            my_dataset_images/
                0_2.png   ← chat 0, message at position 2
                1_0.png   ← chat 1, message at position 0
                ...
        """
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img_dir = path.parent / f"{path.stem}_images"
        with path.open("w", encoding="utf-8") as f:
            for chat_idx, chat in enumerate(self.chat_histories):
                record = chat.to_dict(chat_idx=chat_idx, img_dir=img_dir)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load_from_jsonl(cls, jsonl_path: Union[str, Path]) -> ChatHistoryDataset:
        """
        Load a dataset from a .jsonl file.

        Images are resolved from the sibling ``{stem}_images/`` directory that
        was created by :meth:`save_to_jsonl`.  For text-only datasets the image
        resolution step is a no-op.
        """
        path = Path(jsonl_path)
        base_dir = path.parent
        chat_histories: List[ChatHistory] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chat = ChatHistory.from_dict(json.loads(line))
                    chat.load_images(base_dir)
                    chat_histories.append(chat)
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Malformed record on line {line_no} of {path}: {exc}"
                    ) from exc
        return cls(name=path.stem, chat_histories=chat_histories)

    def pretty_print(self) -> None:
        width = 60
        print(f"{MAGENTA}{'═' * width}")
        print(f" ChatHistoryDataset · {self.name!r} · {len(self)} entries")
        print(f"{'═' * width}{RESET}")
        for i, chat in enumerate(self.chat_histories):
            print(f"{DIM} [{i}]{RESET}")
            chat.pretty_print()
        print(f"{DIM}{'═' * width}{RESET}")

    def append_to_all(
        self,
        content,
        role: Role = Role.USER,
        is_generated: bool = False,
        name_suffix: str = None,
    ) -> None:
        for chat in self.chat_histories:
            chat.append(content, role=role, is_generated=is_generated)
        if name_suffix:
            self.name = self.name + name_suffix

    def prepend_to_all(
        self,
        content,
        role: Role = Role.USER,
        is_generated: bool = False,
        name_suffix: str = None,
    ) -> None:
        for chat in self.chat_histories:
            chat.prepend(content, role=role, is_generated=is_generated)
        if name_suffix:
            self.name = self.name + name_suffix

    def __iter__(self) -> Iterator[ChatHistory]:
        for chat in self.chat_histories:
            yield chat

    def __getitem__(self, item) -> ChatHistory:
        return self.chat_histories[item // self.num_images_per_chat]

    def __len__(self) -> int:
        return len(self.chat_histories) * self.num_images_per_chat

    def __repr__(self) -> str:
        return f"ChatHistoryDataset(name={self.name!r}, size={len(self)})"

    def random_combination_continue(
        self, dataset: ChatHistoryDataset, seed, n_samples: int, name_suffix=None
    ):
        new_dataset = copy.deepcopy(self)
        rng = random.Random(seed)
        incoming_indices = list(range(len(dataset)))
        rng.shuffle(incoming_indices)
        base_indices = list(range(len(self.chat_histories)))
        new_histories = []
        for i in range(n_samples):
            base_idx = base_indices[i % len(base_indices)]
            chat_copy = copy.deepcopy(self.chat_histories[base_idx])
            incoming_idx = incoming_indices[i % len(incoming_indices)]
            for message in dataset[incoming_idx].messages:
                chat_copy.append(content=message.content, role=message.role)
            new_histories.append(chat_copy)
        new_dataset.chat_histories = new_histories
        if name_suffix:
            new_dataset.name += name_suffix
        return new_dataset
