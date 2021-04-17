import json
import pickle
from reranking import re_ranking

m1 = pickle.load(open('mat/mat1_1.pkl', 'rb'))
m2 = pickle.load(open('mat/mat1_2.pkl', 'rb'))
m3 = pickle.load(open('mat/mat1_3.pkl', 'rb'))
ss1, vv1 = m1[0], m1[1]
ss2, vv2 = m2[0], m2[1]
ss3, vv3 = m3[0], m3[1]

zz, vids = m1[2], m1[3]
mat11 = re_ranking(ss1, vv1, 100, 30, 0.8)
mat12 = re_ranking(ss2, vv2, 100, 30, 0.8)
mat13 = re_ranking(ss3, vv3, 100, 30, 0.8)


m1 = pickle.load(open('mat/mat2_1.pkl', 'rb'))
m2 = pickle.load(open('mat/mat2_2.pkl', 'rb'))
m3 = pickle.load(open('mat/mat2_3.pkl', 'rb'))
ss1, vv1 = m1[0], m1[1]
ss2, vv2 = m2[0], m2[1]
ss3, vv3 = m3[0], m3[1]

zz, vids = m1[2], m1[3]
mat21 = re_ranking(ss1, vv1, 100, 30, 0.8)
mat22 = re_ranking(ss2, vv2, 100, 30, 0.8)
mat23 = re_ranking(ss3, vv3, 100, 30, 0.8)

m1 = pickle.load(open('mat/mat3_1.pkl', 'rb'))
m2 = pickle.load(open('mat/mat3_2.pkl', 'rb'))
m3 = pickle.load(open('mat/mat3_3.pkl', 'rb'))
ss1, vv1 = m1[0], m1[1]
ss2, vv2 = m2[0], m2[1]
ss3, vv3 = m3[0], m3[1]

zz, vids = m1[2], m1[3]
mat31 = re_ranking(ss1, vv1, 100, 30, 0.8)
mat32 = re_ranking(ss2, vv2, 100, 30, 0.8)
mat33 = re_ranking(ss3, vv3, 100, 30, 0.8)


mat = mat33 + mat32 + mat31 + mat21 + mat22 + mat23 + mat11 + mat12 + mat13

r = mat.argsort(-1)
res = dict()
for i, sid in enumerate(zz):
    res[sid] = vids[r[i]].tolist()
json.dump(res, open('submit.json', 'w'))