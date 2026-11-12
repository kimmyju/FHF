import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from fhf.labels import MODEL_NAMES
from fhf import v4_likelihood as v4


def cfg():
    return dict(model='', revision='main', seed=42, rank=2, alpha=4, dropout=0.,
                gradient_checkpointing=True, max_length=12, logit_chunk_tokens=3,
                batch_size=2, accumulation=18, eval_batch_size=3, external_epochs=1,
                official_epochs=1, external_per_class=1, external_learning_rate=.01,
                official_learning_rate=.001, weight_decay=0., warmup_ratio=.1, temperatures=[1.,2.])


def tiny(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.set_num_threads(1)
    backend = Tokenizer(WordLevel({'[PAD]':0, '[EOS]':1, '[UNK]':2, 'sample':3,
        **{str(i):i+4 for i in range(25)}}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=29, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, eos_token_id=1))
    path = tmp_path / 'base'
    base.save_pretrained(path)
    tok.save_pretrained(path)
    return base, tok, path


def frame(group):
    return pd.DataFrame([{'id':f'{group}-{i}', 'prompt_id':str(group), 'model':name,
                         'text':f'sample {group} {i}'} for i,name in enumerate(MODEL_NAMES)])


def test_token_alignment_mask_gradients_and_checkpoint_roundtrip(tmp_path):
    base,tok,path = tiny(tmp_path)
    config = cfg()
    model = v4.SourceLM(base,config)
    ids,mask = v4.batch_rows([[3,4,5],[3]],0,model.device)
    model.eval()
    got = model.token_logps(ids,mask,3)
    embed = base.get_input_embeddings()(ids)
    prefix = model.prefix[3][None,None].expand(2,1,-1)
    full_mask = torch.cat([torch.ones(2,1,dtype=mask.dtype),mask],1)
    expected_logits = base(inputs_embeds=torch.cat([prefix,embed],1),attention_mask=full_mask).logits[:,:-1]
    expected = -F.cross_entropy(expected_logits.reshape(-1,29),ids.reshape(-1),reduction='none').reshape_as(ids)
    torch.testing.assert_close(got,expected)
    # Padding must not change the score of the real first token.
    solo = model.token_logps(ids[1:,:1],mask[1:,:1],3)
    torch.testing.assert_close(got[1,0],solo[0,0])
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0.)
    optimizer.zero_grad()
    lp=model.token_logps(ids,mask,3)
    loss=(-(lp*mask).sum(1)/mask.sum(1)).mean()
    loss.backward()
    assert model.prefix.grad[3].abs().sum()>0
    for module in model.modules():
        if isinstance(module,v4.SourceLinear):
            assert module.B.grad[3].abs().sum()>0
            assert module.B.grad[[i for i in range(18) if i!=3]].count_nonzero()==0
    optimizer.step()
    assert all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen)
    model.eval()
    actual=model.token_logps(ids,mask,3).detach()
    assert not torch.equal(actual,model.token_logps(ids,mask,4))
    model.save_adapter(tmp_path/'adapter')
    fresh=v4.build(str(path),config,offline=True)
    fresh.load_adapter(tmp_path/'adapter')
    fresh.eval()
    torch.testing.assert_close(fresh.token_logps(ids,mask,3),actual)


def test_relative_scoring_formula_and_mask():
    logps=torch.tensor([[[-1.,-3.,100.],[-2.,-1.,-100.],[-3.,-2.,0.]]])
    mask=torch.tensor([[1,1,0]])
    got=v4.relative_scores(logps,mask,[1.,2.])
    reference=[]
    for k in range(3):
        reference.append(np.mean([np.exp(-sum(max(float(logps[0,j,t]-logps[0,k,t]),0) for j in range(3))/temp)
                                  for temp in [1.,2.] for t in [0,1]]))
    np.testing.assert_allclose(got.numpy()[0],reference,rtol=1e-6)
    torch.testing.assert_close(got,v4.relative_scores(logps+42,mask,[1.,2.]))


def test_train_and_offline_prediction_end_to_end(tmp_path):
    _,tok,path=tiny(tmp_path)
    config={**cfg(),'model':str(path)}
    tr,va,ex=frame(0),frame(1),frame(2)
    ex['origin']='external'
    ex['source_dataset']='fixture'
    out=tmp_path/'run'
    result=v4.train(tr,va,ex,config,out)
    assert result['evaluation']=='validation_selected_holdout'
    assert len(json.loads((out/'epoch_metrics.json').read_text()))==2
    assert (out/'data_audit.json').is_file()
    history=json.loads((out/'history.json').read_text())
    assert [r['stage'] for r in history]==['external','official']
    scores=v4.predict(out,va,str(path),tmp_path/'submission.parquet',batch_size=2)
    expected=pd.read_csv(out/'predictions.csv')[[f'score::{n}' for n in MODEL_NAMES]].to_numpy()
    np.testing.assert_allclose(scores,expected,atol=1e-6,rtol=1e-5)
    submission=pd.read_parquet(tmp_path/'submission.parquet')
    assert submission.id.tolist()==va.id.tolist()
    assert submission.model.tolist()==np.asarray(MODEL_NAMES)[scores.argmax(1)].tolist()
    with pytest.raises(ValueError,match='Output exists'):
        v4.train(tr,va,ex,config,out)
    with pytest.raises(ValueError,match='overlap'):
        v4.audit_data(tr,tr,ex)


def test_contiguous_truncation():
    class T:
        def __call__(self,text,**kw): return {'input_ids':list(range(int(text)))}
    rows,audit=v4.encode_texts(['9','2'],T(),4)
    assert rows==[[0,1,2,3],[0,1]] and audit['truncated']==1
