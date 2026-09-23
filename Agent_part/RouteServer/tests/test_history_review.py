"""Offline review links preserve parent identity and source/target separation."""

from html.parser import HTMLParser
from urllib.parse import unquote

from mcp_server.data.history_review import write_history_review


def test_review_has_local_media_and_escaped_labels(tmp_path):
    """Parent fragments resolve and original label markup remains text."""
    class Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, tag, attrs):
            self.tags.append((tag, dict(attrs)))

    times = list(range(0, 4_000_000_001, 500_000_000))
    row = {'source_parent_id': 'egowalk:clip<1>', 'split': 'train',
           'observation': {'source_ns': times, 'target_ns': times, 'video_ordinals': list(range(9)),
                           'fpv_paths': [f'media/fpv/{i}.jpg' for i in range(9)], 'bev': None},
           'source_row': {'response': '<script>label</script>'}, 'trajectory_sidecar': None}
    path = write_history_review([row], tmp_path)
    html = path.read_text()
    tags = Tags()
    tags.feed(html)
    images = [attrs['src'] for tag, attrs in tags.tags if tag == 'img']
    assert len(images) == 9
    assert all(url.startswith('../media/fpv/') for url in images)
    assert not any(tag == 'script' for tag, _ in tags.tags)
    link = next(attrs['href'] for tag, attrs in tags.tags if tag == 'a')
    section = next(attrs['id'] for tag, attrs in tags.tags if tag == 'section')
    assert unquote(link[1:]) == section == row['source_parent_id']
    assert '标签适配：待审查' in html
    assert '实际 -4.000s' in html
