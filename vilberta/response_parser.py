from dataclasses import dataclass
from enum import Enum
from collections.abc import Generator

from vilberta.text_section_splitter import (
    StreamSection,
    Part,
    StreamTextSectionSplitter,
)


class SectionType(Enum):
    SPEAK = "speak"
    TEXT = "text"
    TRANSCRIPT = "transcript"


@dataclass
class Section:
    type: SectionType
    content: str


TAG_OPEN = {
    "[speak]": SectionType.SPEAK,
    "[text]": SectionType.TEXT,
    "[transcript]": SectionType.TRANSCRIPT,
}

_SECTIONS = [
    StreamSection("[speak]", "[/speak]", inner_split_on=["\n"]),
    StreamSection("[text]", "[/text]", inner_split_on=None),
    StreamSection("[transcript]", "[/transcript]", inner_split_on=None),
]

_TAG_STRINGS = {s.starting_tag for s in _SECTIONS} | {
    s.ending_tag for s in _SECTIONS if s.ending_tag
}


def parse_response(text: str) -> list[Section]:
    splitter = StreamTextSectionSplitter(sections=_SECTIONS)
    parts = list(splitter.split(text))
    parts.extend(splitter.flush())

    sections: list[Section] = []
    for part in parts:
        if part.text in _TAG_STRINGS or part.section is None:
            continue
        section_type = TAG_OPEN.get(part.section)
        if section_type is None:
            continue
        content = part.text.strip()
        if content:
            sections.append(Section(type=section_type, content=content))
    return sections


class StreamingParser:
    """Incrementally parses streamed LLM chunks into sections.

    Uses StreamTextSectionSplitter under the hood.
    """

    def __init__(self) -> None:
        self._splitter = StreamTextSectionSplitter(sections=_SECTIONS)

    def feed(self, chunk: str) -> Generator[Section, None, None]:
        yield from self._convert(self._splitter.split(chunk))

    def flush(self) -> Generator[Section, None, None]:
        yield from self._convert(self._splitter.flush())

    @staticmethod
    def _convert(parts: Generator[Part, None, None]) -> Generator[Section, None, None]:
        for part in parts:
            if part.text in _TAG_STRINGS or part.section is None:
                continue
            section_type = TAG_OPEN.get(part.section)
            if section_type is None:
                continue
            content = part.text.strip()
            if content:
                yield Section(type=section_type, content=content)
