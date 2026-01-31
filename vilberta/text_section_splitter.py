"""
Stream Text Section Splitter

This module provides utilities for parsing streaming LLM output that is organized
into tagged sections with internal delimiters.

The main use case is when an LLM generates output in a structured format where:
- Different types of content are wrapped in tags (e.g., [speak]...[/speak])
- Within each section, content may need to be split on delimiters (e.g., newlines, periods)
- The output arrives as a stream of chunks, potentially splitting tags or content mid-way

Key features:
- Handles incomplete tags across chunk boundaries (buffers until complete)
- Yields section tags as independent parts
- Splits section content based on configurable delimiters
- Maintains state across streaming chunks (section indices, split indices)
- Supports optional ending tags for sections
- Can flush remaining buffered content at end of stream

Usage:
    section = StreamSection(
        starting_tag="[speak]",
        ending_tag="[/speak]",
        inner_split_on=["\n", ". "]
    )

    splitter = StreamTextSectionSplitter(sections=[section])

    for chunk in stream:
        for part in splitter.split(chunk):
            print(f"Section: {part.section}, Text: {part.text}")

    for part in splitter.flush():
        print(f"Remaining: {part.text}")
"""

from collections.abc import Generator
from dataclasses import dataclass


@dataclass
class StreamSection:
    starting_tag: str
    ending_tag: str | None = None
    inner_split_on: list[str] | None = None


@dataclass
class Part:
    section: str | None
    text: str
    section_index: int
    inner_split_index: int


class StreamTextSectionSplitter:
    def __init__(self, sections: list[StreamSection]):
        self.sections = sections
        self.buffer = ""
        self.current_section: StreamSection | None = None
        self.section_indices: dict[str, int] = {}
        self.current_inner_split_index = 0

    def split(self, chunk: str) -> Generator[Part, None, None]:
        self.buffer += chunk

        while True:
            result = self._process_buffer()
            if result is None:
                break
            yield result

    def flush(self) -> Generator[Part, None, None]:
        if self.buffer:
            part = Part(
                section=self.current_section.starting_tag
                if self.current_section
                else None,
                text=self.buffer,
                section_index=self._get_current_section_index(),
                inner_split_index=self.current_inner_split_index,
            )
            self.buffer = ""
            if self.current_section:
                self.current_inner_split_index += 1
            yield part

    def _process_buffer(self) -> Part | None:
        # Try to find the earliest tag (start or end) in the buffer
        earliest_tag_pos = len(self.buffer)
        found_tag = None
        is_start_tag = False
        matching_section = None

        # Check for start tags
        for section in self.sections:
            pos = self.buffer.find(section.starting_tag)
            if pos != -1 and pos < earliest_tag_pos:
                earliest_tag_pos = pos
                found_tag = section.starting_tag
                is_start_tag = True
                matching_section = section

        # Check for end tags (only if we are currently in a section)
        if (
            self.current_section is not None
            and self.current_section.ending_tag is not None
        ):
            pos = self.buffer.find(self.current_section.ending_tag)
            if pos != -1 and pos < earliest_tag_pos:
                earliest_tag_pos = pos
                found_tag = self.current_section.ending_tag
                is_start_tag = False
                # matching_section must be non-None here since current_section is not None
                matching_section = self.current_section

        # If no tag found, try to process content up to a split point
        if found_tag is None:
            return self._process_content_without_tag()

        # If tag is at position 0, yield the tag itself
        if earliest_tag_pos == 0:
            # At this point, matching_section cannot be None because we found a tag
            assert matching_section is not None
            return self._process_tag(found_tag, is_start_tag, matching_section)

        # There is content before the tag, process it
        return self._process_content_before_tag(earliest_tag_pos)

    def _process_tag(
        self, tag: str, is_start_tag: bool, section: StreamSection
    ) -> Part:
        # Remove tag from buffer
        self.buffer = self.buffer[len(tag) :]

        if is_start_tag:
            # Update section index for this section type
            if section.starting_tag not in self.section_indices:
                self.section_indices[section.starting_tag] = 0
            else:
                self.section_indices[section.starting_tag] += 1

            # Enter new section
            self.current_section = section
            self.current_inner_split_index = 0

            return Part(
                section=None,
                text=tag,
                section_index=self.section_indices[section.starting_tag],
                inner_split_index=0,
            )
        else:
            # Exit current section
            section_tag = (
                self.current_section.starting_tag if self.current_section else None
            )
            section_idx = self._get_current_section_index()
            self.current_section = None
            self.current_inner_split_index = 0

            return Part(
                section=section_tag,
                text=tag,
                section_index=section_idx,
                inner_split_index=0,
            )

    def _process_content_before_tag(self, tag_pos: int) -> Part | None:
        # Extract content before the tag
        content = self.buffer[:tag_pos]

        if self.current_section is None:
            # Outside any section, yield as is
            self.buffer = self.buffer[tag_pos:]
            return Part(
                section=None, text=content, section_index=0, inner_split_index=0
            )

        # Inside a section, try to split by inner delimiters
        return self._split_by_inner_delimiter(content, can_wait=False)

    def _process_content_without_tag(self) -> Part | None:
        if not self.buffer:
            return None

        if self.current_section is None:
            # Outside section and no tag found, need more data
            return None

        # Inside a section, try to split by inner delimiters
        return self._split_by_inner_delimiter(self.buffer, can_wait=True)

    def _split_by_inner_delimiter(self, content: str, can_wait: bool) -> Part | None:
        # Need to check current_section is not None before accessing its attributes
        if self.current_section is None:
            return None

        if not self.current_section.inner_split_on:
            # No inner splitting, cannot yield until we know section is complete
            if can_wait:
                return None
            # Must yield now (tag coming or flushing)
            self.buffer = self.buffer[len(content) :]
            part = Part(
                section=self.current_section.starting_tag,
                text=content,
                section_index=self._get_current_section_index(),
                inner_split_index=self.current_inner_split_index,
            )
            self.current_inner_split_index += 1
            return part

        # Try to find earliest delimiter
        earliest_delim_pos = len(content)
        found_delimiter = None

        for delimiter in self.current_section.inner_split_on:
            pos = content.find(delimiter)
            if pos != -1 and pos < earliest_delim_pos:
                earliest_delim_pos = pos
                found_delimiter = delimiter

        if found_delimiter is None:
            # No delimiter found
            if can_wait:
                # Wait for more data
                return None
            # Must yield now
            self.buffer = self.buffer[len(content) :]
            part = Part(
                section=self.current_section.starting_tag,
                text=content,
                section_index=self._get_current_section_index(),
                inner_split_index=self.current_inner_split_index,
            )
            self.current_inner_split_index += 1
            return part

        # Found a delimiter, split there (include delimiter in the text)
        split_pos = earliest_delim_pos + len(found_delimiter)
        text_to_yield = content[:split_pos]
        self.buffer = self.buffer[split_pos:]

        part = Part(
            section=self.current_section.starting_tag,
            text=text_to_yield,
            section_index=self._get_current_section_index(),
            inner_split_index=self.current_inner_split_index,
        )
        self.current_inner_split_index += 1
        return part

    def _get_current_section_index(self) -> int:
        if self.current_section is None:
            return 0
        return self.section_indices.get(self.current_section.starting_tag, 0)


# stream_text_section_splitter.py


def main() -> None:
    # Test Case 1: Basic section with inner splits
    print("Test 1: Basic section with inner splits")
    section1 = StreamSection(
        starting_tag="[speak]", ending_tag="[/speak]", inner_split_on=["\n", ". "]
    )
    splitter1 = StreamTextSectionSplitter(sections=[section1])

    chunk1 = "[speak]Hello. World\n"
    chunk2 = "Test[/speak]"

    parts1 = list(splitter1.split(chunk1))
    parts1.extend(splitter1.split(chunk2))
    parts1.extend(splitter1.flush())

    assert len(parts1) == 5
    assert parts1[0].text == "[speak]"
    assert parts1[0].section is None
    assert parts1[1].text == "Hello. "
    assert parts1[1].section == "[speak]"
    assert parts1[1].inner_split_index == 0
    assert parts1[2].text == "World\n"
    assert parts1[2].section == "[speak]"
    assert parts1[2].inner_split_index == 1
    assert parts1[3].text == "Test"
    assert parts1[3].section == "[speak]"
    assert parts1[3].inner_split_index == 2
    assert parts1[4].text == "[/speak]"
    assert parts1[4].section == "[speak]"
    print("✓ Passed")

    # Test Case 2: Text outside sections
    print("\nTest 2: Text outside sections")
    section2 = StreamSection(starting_tag="[text]", inner_split_on=None)
    splitter2 = StreamTextSectionSplitter(sections=[section2])

    parts2 = list(splitter2.split("Outside[text]Inside"))
    parts2.extend(splitter2.flush())

    assert len(parts2) == 3
    assert parts2[0].text == "Outside"
    assert parts2[0].section is None
    assert parts2[1].text == "[text]"
    assert parts2[1].section is None
    assert parts2[2].text == "Inside"
    assert parts2[2].section == "[text]"
    print("✓ Passed")

    # Test Case 3: Partial tag across chunks
    print("\nTest 3: Partial tag across chunks")
    section3 = StreamSection(starting_tag="[speak]", inner_split_on=None)
    splitter3 = StreamTextSectionSplitter(sections=[section3])

    parts3 = list(splitter3.split("Test[sp"))
    parts3.extend(splitter3.split("eak]Content"))
    parts3.extend(splitter3.flush())

    assert len(parts3) == 3
    assert parts3[0].text == "Test"
    assert parts3[1].text == "[speak]"
    assert parts3[2].text == "Content"
    print("✓ Passed")

    # Test Case 4: Multiple sections with indices
    print("\nTest 4: Multiple sections with indices")
    section4 = StreamSection(starting_tag="[a]", inner_split_on=[","])
    splitter4 = StreamTextSectionSplitter(sections=[section4])

    parts4 = list(splitter4.split("[a]x,y[a]p,q"))
    parts4.extend(splitter4.flush())

    assert len(parts4) == 6
    assert parts4[0].text == "[a]"
    assert parts4[0].section_index == 0
    assert parts4[1].text == "x,"
    assert parts4[1].section_index == 0
    assert parts4[1].inner_split_index == 0
    assert parts4[2].text == "y"
    assert parts4[2].section_index == 0
    assert parts4[2].inner_split_index == 1
    assert parts4[3].text == "[a]"
    assert parts4[3].section_index == 1
    assert parts4[4].text == "p,"
    assert parts4[4].section_index == 1
    assert parts4[4].inner_split_index == 0
    assert parts4[5].text == "q"
    assert parts4[5].section_index == 1
    assert parts4[5].inner_split_index == 1
    print("✓ Passed")

    # Test Case 5: Empty splits (consecutive delimiters)
    print("\nTest 5: Empty splits")
    section5 = StreamSection(starting_tag="[x]", inner_split_on=[","])
    splitter5 = StreamTextSectionSplitter(sections=[section5])

    parts5 = list(splitter5.split("[x]a,,b"))
    parts5.extend(splitter5.flush())

    assert len(parts5) == 4
    assert parts5[0].text == "[x]"
    assert parts5[1].text == "a,"
    assert parts5[2].text == ","
    assert parts5[3].text == "b"
    print("✓ Passed")

    # Test Case 6: Multiple delimiter types, first match wins
    print("\nTest 6: Multiple delimiters, first match wins")
    section6 = StreamSection(starting_tag="[s]", inner_split_on=[". ", "\n"])
    splitter6 = StreamTextSectionSplitter(sections=[section6])

    parts6 = list(splitter6.split("[s]Hello. World\nTest"))
    parts6.extend(splitter6.flush())

    assert len(parts6) == 4
    assert parts6[0].text == "[s]"
    assert parts6[1].text == "Hello. "
    assert parts6[2].text == "World\n"
    assert parts6[3].text == "Test"
    print("✓ Passed")

    # Test Case 7: Section with no inner split
    print("\nTest 7: Section with no inner split")
    section7 = StreamSection(
        starting_tag="[full]", ending_tag="[/full]", inner_split_on=None
    )
    splitter7 = StreamTextSectionSplitter(sections=[section7])

    parts7 = list(splitter7.split("[full]All this content. No splits\n"))
    parts7.extend(splitter7.split("here[/full]"))
    parts7.extend(splitter7.flush())

    assert len(parts7) == 3
    assert parts7[0].text == "[full]"
    assert parts7[1].text == "All this content. No splits\nhere"
    assert parts7[2].text == "[/full]"
    print("✓ Passed")

    # Test Case 8: Ending tag handling
    print("\nTest 8: Ending tag handling")
    section8 = StreamSection(
        starting_tag="[start]", ending_tag="[end]", inner_split_on=[","]
    )
    splitter8 = StreamTextSectionSplitter(sections=[section8])

    parts8 = list(splitter8.split("[start]a,b[end]outside"))
    parts8.extend(splitter8.flush())

    assert len(parts8) == 5
    assert parts8[0].text == "[start]"
    assert parts8[1].text == "a,"
    assert parts8[2].text == "b"
    assert parts8[3].text == "[end]"
    assert parts8[4].text == "outside"
    print("✓ Passed")

    # Test Case 9: Flush with remaining content
    print("\nTest 9: Flush with remaining content")
    section9 = StreamSection(starting_tag="[s]", inner_split_on=[","])
    splitter9 = StreamTextSectionSplitter(sections=[section9])

    parts9 = list(splitter9.split("[s]incomplete"))
    flushed = list(splitter9.flush())

    assert len(parts9) == 1
    assert parts9[0].text == "[s]"
    assert len(flushed) == 1
    assert flushed[0].text == "incomplete"
    assert flushed[0].section == "[s]"
    print("✓ Passed")

    # Test Case 10: Multiple different sections
    print("\nTest 10: Multiple different section types")
    section10a = StreamSection(starting_tag="[speak]", inner_split_on=[". "])
    section10b = StreamSection(starting_tag="[think]", inner_split_on=["\n"])
    splitter10 = StreamTextSectionSplitter(sections=[section10a, section10b])

    parts10 = list(splitter10.split("[speak]Hi. [think]Thought\n"))
    parts10.extend(splitter10.split("More[speak]End. "))
    parts10.extend(splitter10.flush())

    assert len(parts10) == 7
    assert parts10[0].text == "[speak]"
    assert parts10[0].section_index == 0
    assert parts10[1].text == "Hi. "
    assert parts10[1].section == "[speak]"
    assert parts10[2].text == "[think]"
    assert parts10[2].section_index == 0
    assert parts10[3].text == "Thought\n"
    assert parts10[3].section == "[think]"
    assert parts10[4].text == "More"
    assert parts10[4].section == "[think]"
    assert parts10[5].text == "[speak]"
    assert parts10[5].section_index == 1
    assert parts10[6].text == "End. "
    assert parts10[6].section == "[speak]"
    print("✓ Passed")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
