from render.app import is_calendar_email


def test_calendar_route_matches_first_body_word_case_insensitively():
    assert is_calendar_email("Calendar\nTeam meeting at 10:00")
    assert is_calendar_email("  CALENDAR: Team meeting")


def test_calendar_route_does_not_match_words_containing_calendar():
    assert not is_calendar_email("calendarize this note")
    assert not is_calendar_email("Please calendar this meeting")
