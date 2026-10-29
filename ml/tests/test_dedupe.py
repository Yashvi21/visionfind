from app.search import Hit, format_timestamp, temporal_dedupe

from .conftest import search, upload_ready


def _hit(t, vid="a", sim=0.9, interval=1):
    return Hit(frame_id=int(t * 10), video_id=vid, video_filename=f"{vid}.mp4", timestamp_sec=t,
               similarity=sim, interval_sec=interval)


def test_drops_lower_ranked_neighbours_within_window():
    hits = [_hit(10), _hit(11), _hit(9), _hit(13), _hit(20)]
    assert [h.timestamp_sec for h in temporal_dedupe(hits, 2)] == [10, 13, 20]


def test_window_is_inclusive():
    assert [h.timestamp_sec for h in temporal_dedupe([_hit(10), _hit(12)], 2)] == [10]


def test_zero_window_disables():
    hits = [_hit(10), _hit(11), _hit(12)]
    assert temporal_dedupe(hits, 0) == hits


def test_dedupe_is_per_video():
    hits = [_hit(10, "a"), _hit(10, "b"), _hit(11, "a")]
    assert [(h.video_id, h.timestamp_sec) for h in temporal_dedupe(hits, 2)] == [("a", 10), ("b", 10)]


def test_default_window_is_twice_each_videos_interval():
    hits = [_hit(10, "a", interval=1), _hit(12, "a", interval=1), _hit(13, "a", interval=1),
            _hit(10, "b", interval=5), _hit(20, "b", interval=5), _hit(25, "b", interval=5)]
    kept = [(h.video_id, h.timestamp_sec) for h in temporal_dedupe(hits, None)]
    # a: window 2 -> 12 dropped, 13 kept.  b: window 10 -> 20 dropped (inclusive), 25 kept.
    assert kept == [("a", 10), ("a", 13), ("b", 10), ("b", 25)]


def test_search_endpoint_dedupes(client, video_file):
    upload_ready(client, video_file(10), interval=1)  # frames at 0..9
    # Default window = 2 * 1s. Best is 4; 2,3,5,6 fall within 2s; 1 and 7 survive; 0,8 are within 1s
    # of 1/7 and 9 is within 2s of 7.
    deduped = search(client, "second 4", top_k=10)
    assert deduped[0]["timestamp_sec"] == 4
    assert {r["timestamp_sec"] for r in deduped} == {1, 4, 7}

    everything = search(client, "second 4", top_k=10, dedupe_window_sec=0)
    assert len(everything) == 10
    sims = [r["similarity"] for r in everything]
    assert sims == sorted(sims, reverse=True)


def test_dedupe_still_fills_top_k_when_possible(client, video_file):
    # 12 frames, window 2 -> e.g. 0,3,6,9 style spacing; ask for 4 and expect 4 results.
    upload_ready(client, video_file(12), interval=1)
    assert len(search(client, "second 0", top_k=4)) == 4


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65.9) == "01:05"
    assert format_timestamp(3725) == "1:02:05"
