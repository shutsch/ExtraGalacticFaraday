import nifty8 as ift
import numpy as np




def sample_statistics(self, samples):

        cr=np.array([s for s in samples.iterator(self.ecomponents['chi_red'])])
        mr, vr = samples.sample_stat(self.ecomponents['chi_red'])
        mr=mr.val
        sr=np.sqrt(vr.val)
        #cr_array=cr.val

        ci0=np.array([s for s in samples.iterator(self.ecomponents['chi_int_0'])])
        mi0, vi0 = samples.sample_stat(self.ecomponents['chi_int_0'])
        mi0=mi0.val
        si0=np.sqrt(vi0.val)
        #ci0_array=ci0.val

        cl=np.array([s for s in samples.iterator(self.ecomponents['chi_lum'])])
        ml, vl = samples.sample_stat(self.ecomponents['chi_lum'])
        ml=ml.val
        sl=np.sqrt(vl.val)
        #cl_array=cl.val

        ce0=np.array([s for s in samples.iterator(self.ecomponents['chi_env_0'])])
        me0, ve0 = samples.sample_stat(self.ecomponents['chi_env_0'])
        me0=me0.val
        se0=np.sqrt(ve0.val)
        #ce0_array=ce0.val

        cr_list=[]
        cl_list=[]
        ci0_list=[]
        ce0_list=[]
        for i in range(0,len(cr)):
            cr_list.append(cr[i].val)
            cl_list.append(cl[i].val)
            ci0_list.append(ci0[i].val)
            ce0_list.append(ce0[i].val)
        cr_array=np.array(cr_list)
        cl_array=np.array(cl_list)
        ci0_array=np.array(ci0_list)
        ce0_array=np.array(ce0_list)


        return {'chi_red_mean': mr, 'chi_red_std': sr, 'chi_red_samples': cr_array,
                'chi_int_0_mean': mi0, 'chi_int_0_std': si0, 'chi_int_0_samples': ci0_array,
                'chi_lum_mean': ml, 'chi_lum_std': sl, 'chi_lum_samples': cl_array,
                'chi_env_0_mean': me0, 'chi_env_0_std': se0, 'chi_env_0_samples': ce0_array}