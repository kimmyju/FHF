"""Source-audit never changes labels and never confuses heuristics with provenance."""
import pandas as pd
from fhf.v2_source_audit import audit_selected, save_audit


def test_focus_source_and_conflicting_labels(tmp_path):
    official=pd.DataFrame({
        "id":["t1","t2","t3"],"prompt_id":["p1","p2","p3"],
        "model":["glm-5.3","glm-5.3-flash","claude-fable-5"],
        "text":["official alpha","official beta","official gamma"]})
    selected=pd.DataFrame({
        "id":["e1","e2","e3","e4"],"prompt_id":["q1","q2","q3","q4"],
        "model":["glm-5.3","glm-5.3-flash","claude-fable-5","glm-5.3"],
        "source_dataset":["pool1","pool2","pool1","pool1"],
        "text":["something different","something shared","another text","official gamma"]})
    raw=pd.DataFrame({
        "model":["glm-5.3","claude-fable-5","claude-fable-5"],
        "text":["something different","something shared","another text"],
        "lab":["lab1","lab2","lab1"],
        "flag_kb_proxy":[False,False,False]})
    s, sources, features, rows = audit_selected(official,selected,raw_pool=raw)
    assert s["selected_overlap_official_train"]==1
    assert s["selected_hashes_with_conflicting_labels"]==1
    assert s["raw_pool"]["selected_vs_raw_label_disagreement_hashes"]==1
    assert s["focus"]["glm-5.3"]["selected_rows"]==2
    assert not "text" in rows
    save_audit((s,sources,features,rows),tmp_path)
    assert (tmp_path/"summary.json").is_file()


def test_text_is_explicit_opt_in():
    official=pd.DataFrame({
        "id":["t1","t2","t3"],"prompt_id":["p1","p2","p3"],
        "model":["glm-5.3","glm-5.3-flash","claude-fable-5"],
        "text":["official one","official two","official three"]})
    selected=pd.DataFrame({
        "id":["e1","e2","e3"],"prompt_id":["q1","q2","q3"],
        "model":["glm-5.3","glm-5.3-flash","claude-fable-5"],
        "source_dataset":["pool"]*3,
        "text":["external one","external two","external three"]})
    s, _, _, rows = audit_selected(official,selected,include_text=True)
    assert "text" in rows
