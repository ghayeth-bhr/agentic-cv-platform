"""Slot-filling dialogue for the labeling agent.

The agent turns a conversation into a complete task specification:

* which assets to label,
* detection or segmentation,
* the object classes (each with optional visual attributes and SAM 3 prompts),
* the export format.

Extraction can use a local LLM, but **code decides** what is missing and what to
ask: an extractor proposes a :class:`DraftPatch`, :func:`apply_patch` merges it,
:func:`missing_slots` checks the result, and :func:`next_question` asks one
templated question with quick replies. A 4B local model only has to fill a JSON
schema, never to run the conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from .contracts import TaskSpec

TaskType = Literal["detection", "segmentation"]
ExportFormat = Literal["coco", "yolo"]
Slot = Literal["assets", "classes", "task_type"]
ClassesMode = Literal["replace", "append", "remove"]

REQUIRED_SLOTS: tuple[Slot, ...] = ("assets", "classes", "task_type")
MAX_CLASSES = 10
MAX_TASK_TYPE_QUESTIONS = 1

_SEGMENTATION_HINTS = re.compile(
    r"\b(segment\w*|masks?|polygons?|outlines?|contours?|pixel\w*|instance seg\w*)\b", re.I
)
_DETECTION_HINTS = re.compile(
    r"\b(detect\w*|bounding|b?boxe?s?|bbox\w*|locat\w*|count\w*)\b", re.I
)
_YOLO = re.compile(r"\byolo\w*\b", re.I)
_COCO = re.compile(r"\bcoco\b", re.I)
_APPEND = re.compile(r"^\s*(?:please\s+)?(?:(?:also|add|plus|and|include)\b[\s:,]*)+", re.I)
_REMOVE = re.compile(
    r"^\s*(?:please\s+)?(?:remove|drop|delete|exclude|without|no(?:t)?(?: more)?)\b[\s:,]*", re.I
)
# Every optional word ends in \b: without it "for" matched inside "forklifts"
# and "a" inside "apples".
_LEAD = re.compile(
    r"^\s*(?:(?:hi|hello|hey)\b)?[\s,!.]*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?"
    r"(?:i\s+(?:want|need|would like)\s+(?:you\s+)?to\s+|let'?s\s+|(?:i\s+(?:want|need)|give\s+me|get\s+me)\b\s*)?"
    r"(?:(?:(?:bounding\s+)?boxes|masks|polygons|outlines|labels|annotations)\s+(?:of|for|around|on)\b\s*)?"
    r"(?:(?:create|build|make|generate|produce|prepare|label|annotate|segment|detect|find|locate|count)\b)?\s*"
    r"(?:me\b\s*)?(?:(?:a|an|the)\b)?\s*"
    r"(?:new\b\s*)?(?:(?:object\s+)?(?:detection|segmentation|instance segmentation)\b\s*)?"
    r"(?:(?:dataset|data\s*set|annotations?|labels?)\b)?\s*"
    r"(?:(?:of|for|with|showing|containing|on|that\s+(?:labels|detects|segments))\b)?\s*",
    re.I,
)
_CLASS_CUE = re.compile(
    r"\b(label\w*|annotat\w*|segment\w*|detect\w*|find|locate|count|dataset|class(?:es)?|"
    r"objects?|only|instead|actually|rather|change|switch|replace)\b",
    re.I,
)
_TRAILING = [
    re.compile(r"[\s,;]*(?:and\s+)?(?:then\s+)?(?:export\w*|download\w*|save\w*|format\w*)\b.*$", re.I),
    re.compile(r"[\s,;]*\b(?:in|as|to)\s+(?:yolo|coco)\b.*$", re.I),
    re.compile(r"[\s,;]*\b(?:using|with)\s+(?:bounding\s+)?(?:boxes|masks|polygons|sam\s*3?)\b.*$", re.I),
    re.compile(r"[\s,;]*\bfor\s+(?:object\s+)?(?:detection|segmentation)\b.*$", re.I),
    re.compile(r"[\s,;]*\b(?:on|from|in)\s+(?:(?:the|these|this|my|our|selected|all)\s+)+(?:\w+\s+){0,2}(?:assets?|images?|photos?|pictures?|videos?|frames?|footage|clips?|recordings?|data(?:set)?)\b.*$", re.I),
    re.compile(r"[\s,;]*\b(?:please|thanks|thank you)\b.*$", re.I),
    re.compile(r"[\s,;]*\b(?:yolo|coco)(?:\s+format)?\s*$", re.I),
    re.compile(r"\s+(?:(?:detection|segmentation)\s+)?(?:dataset|data\s*set|labels|annotations)\s*$", re.I),
]
_LEAD_ONLY_TASK = re.compile(
    r"^\s*(?:(?:object\s+)?detection|segmentation|instance segmentation|boxes|masks|bounding boxes)\s*[.!]?\s*$",
    re.I,
)
_RELATION = re.compile(r"\b(with|wearing|in|on|holding|without|near|carrying|inside|under|of)\b", re.I)
_GENERIC = {
    "a", "an", "the", "my", "our", "your", "this", "these", "that", "it", "them",
    "project", "dataset", "datasets", "data", "set", "model", "training", "train",
    "images", "image", "photos", "photo", "pictures", "picture", "videos", "video",
    "frames", "assets", "asset", "something", "stuff", "things", "objects", "object",
    "task", "new", "some", "please", "thanks", "thank", "you", "ok", "okay", "yes",
    "no", "sure", "computer", "vision", "cv", "ai", "for", "me", "us", "all",
    "go", "ahead", "continue", "start", "proceed", "done", "fine", "great", "good",
    "looks", "perfect", "right", "correct", "that", "is", "s", "lets", "let", "run",
    "now", "again", "same",
}
_SPLIT = re.compile(r"\s*(?:,|;|/|&|\band\b|\bor\b|\+)\s*", re.I)


@dataclass(frozen=True, kw_only=True)
class ClassSpec:
    name: str
    attributes: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attributes": list(self.attributes),
            "prompts": list(self.prompts),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClassSpec":
        return cls(
            name=str(value["name"]),
            attributes=tuple(value.get("attributes", ())),
            prompts=tuple(value.get("prompts", ())),
        )

    @property
    def prompt(self) -> str:
        return self.prompts[0] if self.prompts else rules_prompt(self)


@dataclass(frozen=True, kw_only=True)
class TaskDraft:
    asset_ids: tuple[str, ...] = ()
    task_type: TaskType | None = None
    classes: tuple[ClassSpec, ...] = ()
    export_format: ExportFormat | None = None
    sample_size: int = 5
    max_iterations: int = 5
    pending_slot: Slot | None = None
    questions: dict[str, int] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_ids": list(self.asset_ids),
            "task_type": self.task_type,
            "classes": [item.to_dict() for item in self.classes],
            "export_format": self.export_format,
            "sample_size": self.sample_size,
            "max_iterations": self.max_iterations,
            "pending_slot": self.pending_slot,
            "questions": dict(self.questions),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskDraft":
        return cls(
            asset_ids=tuple(str(item) for item in value.get("asset_ids", ())),
            task_type=value.get("task_type"),
            classes=tuple(ClassSpec.from_dict(item) for item in value.get("classes", ())),
            export_format=value.get("export_format"),
            sample_size=int(value.get("sample_size", 5)),
            max_iterations=int(value.get("max_iterations", 5)),
            pending_slot=value.get("pending_slot"),
            questions=dict(value.get("questions", {})),
            assumptions=tuple(value.get("assumptions", ())),
        )

    @property
    def complete(self) -> bool:
        return not missing_slots(self)

    def task_spec(self) -> TaskSpec:
        """The v1 contract kept for the calibration loop and older clients."""

        if not self.classes or self.task_type is None:
            raise ValueError("draft is incomplete")
        attributes: list[str] = []
        for item in self.classes:
            for attribute in item.attributes:
                if attribute not in attributes:
                    attributes.append(attribute)
        return TaskSpec(
            target_objects=tuple(item.name for item in self.classes),
            task_type=self.task_type,
            visual_attributes=tuple(attributes),
            sample_size=self.sample_size,
            max_iterations=self.max_iterations,
        )


@dataclass(frozen=True, kw_only=True)
class DraftPatch:
    """What one user turn says. ``None`` means "not mentioned"."""

    task_type: TaskType | None = None
    classes: tuple[ClassSpec, ...] | None = None
    classes_mode: ClassesMode = "replace"
    export_format: ExportFormat | None = None
    sample_size: int | None = None
    max_iterations: int | None = None


@dataclass(frozen=True, kw_only=True)
class ExtractionResult:
    patch: DraftPatch
    mode: str  # "ollama" | "rules"
    note: str | None = None


class DraftExtractor(Protocol):
    def extract(
        self, message: str, *, draft: TaskDraft, pending_slot: Slot | None
    ) -> ExtractionResult: ...


@dataclass(frozen=True, kw_only=True)
class Question:
    slot: Slot
    text: str
    quick_replies: tuple[str, ...] = ()


# --------------------------------------------------------------------------- rules


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _is_generic(phrase: str) -> bool:
    words = _words(phrase)
    return not words or all(word in _GENERIC for word in words)


def _only_task_words(phrase: str) -> bool:
    """"boxes please" names the task, not a class; "cardboard boxes" is a class."""

    words = _words(phrase)
    meaningful = [
        word for word in words
        if word not in _GENERIC
        and not _SEGMENTATION_HINTS.fullmatch(word)
        and not _DETECTION_HINTS.fullmatch(word)
        and word not in {"yolo", "coco", "format", "bounding", "instance"}
    ]
    return not meaningful


def clean_class_phrase(text: str) -> str:
    phrase = re.sub(r"\s+", " ", text or "").strip()
    for _ in range(2):
        phrase = _LEAD.sub("", phrase, count=1)
        for pattern in _TRAILING:
            phrase = pattern.sub("", phrase)
    phrase = re.sub(r"^(?:all|every|each|any)\s+", "", phrase, flags=re.I)
    return phrase.strip(" .,:;!?-\"'“”")


def split_classes(phrase: str) -> tuple[str, ...]:
    """Split "helmets, vests and gloves" into classes.

    A phrase with a relation ("workers wearing yellow vests") is one class:
    splitting it would change what the user asked for.
    """

    phrase = phrase.strip()
    if not phrase:
        return ()
    if _RELATION.search(phrase):
        parts = [phrase]
    else:
        parts = [part for part in _SPLIT.split(phrase) if part]
    seen: list[str] = []
    for part in parts:
        cleaned = clean_class_phrase(part)
        if (
            cleaned
            and not _is_generic(cleaned)
            and not _only_task_words(cleaned)
            and cleaned.lower() not in {item.lower() for item in seen}
        ):
            seen.append(cleaned)
    return tuple(seen[:MAX_CLASSES])


def detect_task_type(message: str) -> TaskType | None:
    segmentation = bool(_SEGMENTATION_HINTS.search(message))
    detection = bool(_DETECTION_HINTS.search(message))
    if segmentation and not detection:
        return "segmentation"
    if detection and not segmentation:
        return "detection"
    if segmentation and detection:
        # "segmentation masks, not boxes" vs "boxes not masks": the first hint wins.
        seg_at = _SEGMENTATION_HINTS.search(message).start()  # type: ignore[union-attr]
        det_at = _DETECTION_HINTS.search(message).start()  # type: ignore[union-attr]
        return "segmentation" if seg_at < det_at else "detection"
    return None


def detect_export_format(message: str) -> ExportFormat | None:
    if _YOLO.search(message):
        return "yolo"
    if _COCO.search(message):
        return "coco"
    return None


class RulesDraftExtractor:
    """Deterministic extractor used offline and as the LLM fallback."""

    def extract(
        self, message: str, *, draft: TaskDraft, pending_slot: Slot | None
    ) -> ExtractionResult:
        text = re.sub(r"\s+", " ", message or "").strip()
        task_type = detect_task_type(text)
        export_format = detect_export_format(text)
        mode: ClassesMode = "replace"
        body = text
        if _REMOVE.match(body):
            mode = "remove"
            body = _REMOVE.sub("", body, count=1)
        elif _APPEND.match(body) and draft.classes:
            mode = "append"
            body = _APPEND.sub("", body, count=1)

        classes: tuple[ClassSpec, ...] | None = None
        if not _LEAD_ONLY_TASK.match(body):
            names = split_classes(clean_class_phrase(body))
            if names:
                classes = tuple(ClassSpec(name=name) for name in names)
        if classes is None and mode != "replace":
            mode = "replace"
        if classes is not None and task_type is not None:
            # "cardboard boxes" is a class, not a request for bounding boxes.
            remainder = text
            for item in classes:
                remainder = re.sub(re.escape(item.name), " ", remainder, flags=re.I)
            task_type = detect_task_type(remainder)
        if (
            classes is not None
            and mode == "replace"
            and draft.classes
            and pending_slot != "classes"
            and not _CLASS_CUE.search(text)
        ):
            # Once classes exist, replacing them needs a labeling cue ("segment
            # X", "only Y", "label Z instead"). Answers like "Segmentation
            # please" or "go ahead" must not overwrite the class list.
            classes = None
        return ExtractionResult(
            patch=DraftPatch(
                task_type=task_type,
                classes=classes,
                classes_mode=mode,
                export_format=export_format,
            ),
            mode="rules",
        )


# --------------------------------------------------------------------------- merge


def apply_patch(draft: TaskDraft, patch: DraftPatch) -> TaskDraft:
    classes = draft.classes
    if patch.classes is not None:
        if patch.classes_mode == "append":
            known = {item.name.lower() for item in classes}
            classes = classes + tuple(item for item in patch.classes if item.name.lower() not in known)
        elif patch.classes_mode == "remove":
            drop = {item.name.lower() for item in patch.classes}
            classes = tuple(
                item for item in classes
                if item.name.lower() not in drop
                and not any(name in item.name.lower() or item.name.lower() in name for name in drop)
            )
        else:
            classes = patch.classes
    return replace(
        draft,
        task_type=patch.task_type or draft.task_type,
        classes=classes[:MAX_CLASSES],
        export_format=patch.export_format or draft.export_format,
        sample_size=patch.sample_size or draft.sample_size,
        max_iterations=patch.max_iterations or draft.max_iterations,
    )


def missing_slots(draft: TaskDraft) -> tuple[Slot, ...]:
    missing: list[Slot] = []
    if not draft.asset_ids:
        missing.append("assets")
    if not draft.classes:
        missing.append("classes")
    if draft.task_type is None:
        missing.append("task_type")
    return tuple(missing)


def resolve_defaults(draft: TaskDraft) -> TaskDraft:
    """Fill slots that may be assumed instead of asked, and record why."""

    assumptions = list(draft.assumptions)
    task_type = draft.task_type
    if (
        task_type is None
        and draft.classes
        and draft.asset_ids
        and draft.questions.get("task_type", 0) >= MAX_TASK_TYPE_QUESTIONS
    ):
        task_type = "detection"
        assumptions.append("Assumed detection (bounding boxes); say “segmentation” to change it.")
    export_format = draft.export_format
    if export_format is None and task_type is not None and draft.classes and draft.asset_ids:
        export_format = "coco"
        assumptions.append("Exporting COCO by default; say “YOLO” to change it.")
    return replace(
        draft,
        task_type=task_type,
        export_format=export_format,
        assumptions=tuple(dict.fromkeys(assumptions)),
    )


def next_question(draft: TaskDraft) -> Question | None:
    missing = missing_slots(draft)
    if not missing:
        return None
    slot = missing[0]
    if slot == "assets":
        return Question(
            slot="assets",
            text=(
                "Which assets should I label? Select one or more in the asset list, "
                "then send your message again."
            ),
        )
    if slot == "classes":
        if draft.questions.get("classes", 0):
            text = (
                "I still need the object classes. Name each thing to label, separated by "
                "commas — for example “hard hats, safety vests”."
            )
        else:
            text = (
                "What should I label? List the object classes, separated by commas — "
                "for example “hard hats, safety vests”. Add details like colors if they matter."
            )
        return Question(slot="classes", text=text)
    return Question(
        slot="task_type",
        text=(
            f"Should I draw bounding boxes (detection) or pixel masks (segmentation) for "
            f"{join_names(draft.classes)}?"
        ),
        quick_replies=("Detection", "Segmentation"),
    )


def mark_asked(draft: TaskDraft, question: Question | None) -> TaskDraft:
    if question is None:
        return replace(draft, pending_slot=None)
    questions = dict(draft.questions)
    questions[question.slot] = questions.get(question.slot, 0) + 1
    return replace(draft, pending_slot=question.slot, questions=questions)


# --------------------------------------------------------------------------- prompts


def rules_prompt(item: ClassSpec) -> str:
    """Short SAM 3 noun phrase: the class plus attributes it does not already contain."""

    name = item.name.strip()
    lowered = name.lower()
    attributes = [
        attribute.strip()
        for attribute in item.attributes
        if attribute.strip() and attribute.strip().lower() not in lowered
    ]
    if not attributes:
        return name or "object"
    return f"{name} with {', '.join(attributes)}"


def join_names(classes: tuple[ClassSpec, ...]) -> str:
    names = [f"“{item.name}”" for item in classes]
    if len(names) <= 1:
        return names[0] if names else "the objects"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def draft_from_spec(spec: TaskSpec, *, asset_ids: tuple[str, ...]) -> TaskDraft:
    """Lift a v1 TaskSpec (planner output) into a draft: one class per target."""

    if len(spec.target_objects) == 1:
        classes = (ClassSpec(name=spec.target_objects[0], attributes=spec.visual_attributes),)
    else:
        classes = tuple(ClassSpec(name=name) for name in spec.target_objects)
        if spec.visual_attributes:
            classes = tuple(replace(item, attributes=spec.visual_attributes) for item in classes)
    return TaskDraft(
        asset_ids=asset_ids,
        task_type=spec.task_type,
        classes=classes,
        sample_size=spec.sample_size,
        max_iterations=spec.max_iterations,
    )
