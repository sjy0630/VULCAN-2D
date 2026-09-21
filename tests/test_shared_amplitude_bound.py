import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from check_shared_amplitude_bound import common_shape_floor,relaxed_interval_floor


class SharedAmplitudeBoundTests(unittest.TestCase):
    def test_closed_form_floor_matches_unrestricted_linear_least_squares(self):
        rng=np.random.default_rng(318);n,m=5,7;yp=rng.normal(size=(n,m));yn=rng.normal(size=(n,m))
        matrix=[];target=[]
        for branch,data in enumerate([yp,yn]):
            for i in range(n):
                for j in range(m):
                    row=np.zeros(n+2*m);row[i]=1;row[n+branch*m+j]=1
                    matrix.append(row);target.append(data[i,j])
        matrix=np.array(matrix);target=np.array(target)
        pars=np.linalg.lstsq(matrix,target,rcond=None)[0]
        expected=np.sqrt(np.mean((matrix@pars-target)**2))
        actual=common_shape_floor(yp,yn)
        self.assertAlmostEqual(actual['exact_arbitrary_common_shapes_rmse_decade'],expected,places=12)
        self.assertLessEqual(actual['ratio_only_lower_bound_rmse_decade'],expected)

    def test_separable_curves_have_zero_floor_and_amplitude_shifts_do_not_help(self):
        v=np.linspace(0,1,9);a=np.arange(5)[:,None];yp=a+v;yn=a+2*v-.4
        self.assertLess(common_shape_floor(yp,yn)['exact_arbitrary_common_shapes_rmse_decade'],1e-14)
        yn=yn+np.arange(5)[:,None]*.2
        before=common_shape_floor(yp,yn);shift=np.arange(5)[:,None]**2
        after=common_shape_floor(yp+shift,yn+shift)
        self.assertAlmostEqual(before['ratio_only_lower_bound_rmse_decade'],after['ratio_only_lower_bound_rmse_decade'],places=12)

    def test_interval_relaxation_is_conservative(self):
        values=np.array([[-2.,-1.],[-.1,.1],[0.,.2]])
        exact=np.sqrt(np.mean((values-values.mean(axis=0))**2))/2
        self.assertAlmostEqual(relaxed_interval_floor(values,values),exact,places=6)
        self.assertLess(relaxed_interval_floor(values-.1,values+.1),exact)
        self.assertLess(relaxed_interval_floor(values-5,values+5),1e-7)


if __name__=='__main__':unittest.main()
