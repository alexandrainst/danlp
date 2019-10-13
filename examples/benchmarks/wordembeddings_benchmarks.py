from danlp.datasets import WordSim353Da
from danlp.models.embeddings import AVAILABLE_EMBEDDINGS, load_wv_with_gensim
import tabulate


def load_wv_models():
    for da_wv_model in AVAILABLE_EMBEDDINGS:
        yield da_wv_model, load_wv_with_gensim(da_wv_model)


ws353 = WordSim353Da()
dsd_path = "/home/alexandra/Data/gold_sims_da.csv"

data = []

for model_name, wv in load_wv_models():
    correlation_on_dsd = wv.evaluate_word_pairs(dsd_path, delimiter="\t")
    spearman_rho_dsd = correlation_on_dsd[1].correlation
    oov_dsd = correlation_on_dsd[2]

    correlation_on_ws353 = wv.evaluate_word_pairs(ws353.file_path, delimiter=',')
    spearman_rho_ws353 = correlation_on_ws353[1].correlation
    oov_ws353 = correlation_on_ws353[2]

    data.append([model_name, len(wv.vocab), wv.vector_size, spearman_rho_ws353, oov_ws353, spearman_rho_dsd, oov_dsd])

headers = ['Model', 'Vocab', 'Vec Size', 'WS353-rho', 'WS353-OOV', 'DSD-rho', 'DSD-OOV']
aligns = ['left', 'center', 'center', 'center', 'center', 'center', 'center']

print(tabulate.tabulate(data, headers=headers, tablefmt='github', colalign=aligns))

# Outputs:
# | Model         |  Vocab  |  Vec Size  |  WS353-rho  |  WS353-OOV  |  DSD-rho  |  DSD-OOV  |
# |---------------|---------|------------|-------------|-------------|-----------|-----------|
# | wiki.da.wv    | 312956  |    300     |  0.638902   |   0.84986   |  0.20488  |  1.0101   |
# | cc.da.wv      | 2000000 |    300     |  0.532651   |   1.69972   | 0.313191  |     0     |
# | conll17.da.wv | 1655870 |    100     |  0.548538   |   1.69972   | 0.149638  |     0     |
# | news.da.wv    | 2404836 |    300     |  0.540629   |   4.24929   | 0.306499  |     0     |
# | sketch_engine | 2722811 |    100     |  0.595543   |   0.84986   | 0.196315  |     0     |
