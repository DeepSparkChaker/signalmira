from __future__ import annotations
import json
import runpy
from types import SimpleNamespace


def test_evaluation_prints_json_and_confusions(tmp_path,capsys):
    path=tmp_path/'gold.jsonl'
    path.write_text('\n'.join([json.dumps({'language':'en','text':'hello'}) ,
                                json.dumps({'language':'es','text':'hola'})])+'\n')
    import pathlib
    mod=runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1]/'scripts/evaluate_language.py'))
    evaluator=mod['evaluate']
    evaluator.__globals__['detect_text']=lambda text: SimpleNamespace(language='en')
    evaluator(path)
    output=json.loads(capsys.readouterr().out)
    assert output['n']==2
    assert output['accuracy']==0.5
    assert output['confusions']=={'es->en':1}
