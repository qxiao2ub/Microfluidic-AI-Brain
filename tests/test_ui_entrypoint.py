"""Render-path smoke check with a lightweight Streamlit API stand-in.

Does not replace an actual hosted Streamlit integration test. It checks that
all tabs and click-handler code can execute using the bundled data.
"""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FakeStreamlit(types.ModuleType):
    def __init__(self, exercise_actions: bool):
        super().__init__("streamlit")
        self.exercise_actions = exercise_actions
        self.sidebar = self
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cache_resource(self, **kwargs):
        return lambda func: func

    def columns(self, count, **kwargs):
        return [self] * (count if isinstance(count, int) else len(count))

    def tabs(self, labels):
        return [self] * len(labels)

    def form(self, *args, **kwargs):
        return self

    def expander(self, *args, **kwargs):
        return self

    def file_uploader(self, *args, **kwargs):
        if self.exercise_actions and kwargs.get('key') == 'batch_upload':
            data = (ROOT / 'data' / 'sample_batch_input.csv').read_bytes()
            return types.SimpleNamespace(name='sample_batch_input.csv', getvalue=lambda: data)
        return None

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index]

    def number_input(self, *args, **kwargs):
        return kwargs.get('value', 0.0)

    def slider(self, *args, **kwargs):
        return kwargs.get('value', kwargs.get('min_value', 0))

    def form_submit_button(self, *args, **kwargs):
        return self.exercise_actions

    def button(self, *args, **kwargs):
        return self.exercise_actions

    def stop(self):
        raise AssertionError('The app stopped unexpectedly during smoke rendering')

    def error(self, message, **kwargs):
        self.messages.append(message)
        raise AssertionError('Streamlit displayed an error: ' + str(message))

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


def _smoke(exercise_actions: bool):
    fake = FakeStreamlit(exercise_actions)
    previous = sys.modules.get('streamlit')
    sys.modules['streamlit'] = fake
    name = 'streamlit_smoke_actions' if exercise_actions else 'streamlit_smoke_default'
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT/'app.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert module.bundle.metrics.shape[0] == 3
        assert module.bundle.cleaned_data.shape[0] == 868
    finally:
        if previous is None:
            sys.modules.pop('streamlit', None)
        else:
            sys.modules['streamlit'] = previous
        sys.modules.pop(name, None)


def test_render_all_tabs_without_clicks():
    _smoke(False)


def test_render_single_batch_and_inverse_actions():
    _smoke(True)
