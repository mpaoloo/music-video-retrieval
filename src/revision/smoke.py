"""Быстрая проверка обучения на искусственных данных до долгой загрузки."""
import numpy as np
import torch
from transformers import ASTConfig, ASTForAudioClassification
from .common import config
from .train import fit, score, normalize_fit, standardize
from .metrics import ranks_from_scores, summarize
from .features import gates, semantic_warning


def main():
    torch.set_num_threads(2)
    cfg=config();cfg.update(epochs=50,patience=50,embedding_dim=8,batch_size=32,learning_rate=.01)
    rng=np.random.default_rng(7)
    x=rng.normal(size=(64,12)).astype('float32')
    audio=x.copy()
    tr=np.arange(48);val=np.arange(48,64)
    model,v,a,vs,aus,history,_=fit(x,audio,tr,val,cfg,42,torch.device('cpu'))
    result=summarize(ranks_from_scores(score(model,v[val],a[val],torch.device('cpu'))))
    assert result['hit1']>=.75,result
    assert history[-1]['train_loss']<history[0]['train_loss'],history
    np.testing.assert_allclose(vs[0],x[tr].mean(0))
    assert gates([.7,.8,.1],[.1,.1,.1],cfg)[0]
    assert not gates([.7,.8,.9],[.9,.9,.9],cfg)[1]
    assert semantic_warning('A mismatch in mood. The rhythm clashes.')
    assert not semantic_warning('The music supports the scene.')
    # Проверяем тот же API AST, что будет вызван с предобученными весами.
    small=ASTConfig(hidden_size=24,num_hidden_layers=1,num_attention_heads=3,
                    intermediate_size=48,num_mel_bins=32,max_length=32,patch_size=8,
                    frequency_stride=8,time_stride=8,num_labels=3)
    ast=ASTForAudioClassification(small).eval()
    with torch.no_grad():
        result=ast.audio_spectrogram_transformer(input_values=torch.randn(1,32,32))
        assert result.pooler_output.shape==(1,24)
        assert ast.classifier(result.pooler_output).shape==(1,3)
    print('SMOKE_OK: paired training, train-only scaling, audio gates, AST forward',flush=True)


if __name__=='__main__':main()
