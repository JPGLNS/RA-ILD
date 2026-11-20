import unittest
from ra_ild_trb.layered_experiments import *
class LayeredTests(unittest.TestCase):
 def test_127_and_1905(self):
  self.assertEqual(len(build_layer1_combinations()),127)
  self.assertEqual(len(build_layer1_manifest()),1905)
 def test_rank_by_cohort_algorithm(self):
  rows=[]
  for c in COHORTS:
   for a in ALGORITHMS:
    for i in range(7):
     rows.append(dict(cohort=c,algorithm=a,combination_id=f"C{i}",roc_auc=.7+i/100,pr_auc=.6+i/100,n_repeats=100))
  selected=rank_layer1_results(rows,top_n=5)
  self.assertEqual(len(selected),75)
 def test_layer2_four_states_and_threshold(self):
  selected=[dict(cohort="total",algorithm="elastic_net",rank=1,combination_id="C1",feature_groups=[CORE83_GROUPS[0]])]
  rows=build_layer2_manifest(selected)
  self.assertEqual(len(rows),4)
  ra=next(x for x in rows if x["include_ra_hit"])
  self.assertEqual((ra["ra_hit_threshold_pct"],ra["ra_hit_delta_pct"]),(20.,10.))
if __name__=="__main__": unittest.main()
