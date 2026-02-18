from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class DomainData:
    """ """

    words: list[str] = field(default_factory=list)
    spaces: list[str] = field(default_factory=list)
    sent_ids: list[int] = field(default_factory=list)
    par_ids: list[int] = field(default_factory=list)
    sent_to_par_ids: dict[int, int] = field(default_factory=dict)
    par_to_sent_ids: dict[int, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    sent_to_word_ids: dict[int, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    par_to_word_ids: dict[int, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class AlignMaps:
    """ """

    src_to_tgt: dict[int, list[int]] = field(default_factory=dict)
    tgt_to_src: dict[int, list[int]] = field(default_factory=dict)


@dataclass
class AlignmentData:
    """ """

    src: DomainData = field(default_factory=DomainData)
    tgt: DomainData = field(default_factory=DomainData)
    align: AlignMaps = field(default_factory=AlignMaps)