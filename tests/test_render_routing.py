from render.app import calendar_event_title, is_calendar_email


def test_calendar_route_matches_subject_case_insensitively():
    assert is_calendar_email("Calendar: Team meeting")
    assert is_calendar_email("Team meeting - CALENDAR")


def test_calendar_route_does_not_match_words_containing_calendar():
    assert not is_calendar_email("calendarize this note")
    assert not is_calendar_email("Team meeting")


def test_calendar_word_is_removed_from_event_title():
    assert calendar_event_title("  calendar  Team meeting  ") == "Team meeting"
    assert calendar_event_title("Team calendar meeting") == "Team meeting"
