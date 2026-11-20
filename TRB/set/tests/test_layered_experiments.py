import unittest
from ra_ild_trb.layered_experiments import *

class LayeredTests(unittest.TestCase):
 def test_127_and_381(self):
  self.assertEqual(len(build_layer1_combinations()),127)
  manifest=build_layer1_manifest()
  self.assertEqual(len(manifest),381)
  self.assertEqual({x["algorithm"] for x in manifest},{"elastic_net"})
  for cohort in COHORTS:
   self.assertEqual(sum(x["cohort"]==cohort for x in manifest),127)

 def test_rank_top5_per_cohort(self):
  rows=[]
  for cohort in COHORTS:
   for i in range(7):
    rows.append(dict(cohort=cohort,algorithm="elastic_net",combination_id=f"C{i}",
                     roc_auc=.7+i/100,pr_auc=.6+i/100,n_repeats=100,
                     feature_groups=[CORE83_GROUPS[0]]))
  selected=rank_layer1_results(rows,top_n=5)
  self.assertEqual(len(selected),15)
  self.assertEqual({x["algorithm"] for x in selected},{"elastic_net"})
  for cohort in COHORTS:
   self.assertEqual(sum(x["cohort"]==cohort for x in selected),5)

 def test_reject_non_elastic_net_results(self):
  row=dict(cohort="total",algorithm="ridge",combination_id="C1",
           roc_auc=.8,pr_auc=.7,n_repeats=100)
  with self.assertRaises(ValueError):
   rank_layer1_results([row])

 def test_layer2_four_states_and_threshold(self):
  selected=[dict(cohort="total",algorithm="elastic_net",rank=1,
                 combination_id="C1",feature_groups=[CORE83_GROUPS[0]])]
  rows=build_layer2_manifest(selected)
  self.assertEqual(len(rows),4)
  self.assertEqual({x["algorithm"] for x in rows},{"elastic_net"})
  ra=next(x for x in rows if x["include_ra_hit"])
  self.assertEqual((ra["ra_hit_threshold_pct"],ra["ra_hit_delta_pct"]),(20.,10.))

if __name__=="__main__": unittest.main()
