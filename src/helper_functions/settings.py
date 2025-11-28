import numpy as np
import nifty8 as ift



def gal_settings(lgrm, g_rm, g_rm_err):
    gal_data_domain = ift.makeDomain(ift.UnstructuredDomain((lgrm,)))

    gal_rm = ift.Field(gal_data_domain, g_rm)
    gal_stddev = ift.Field(gal_data_domain, g_rm_err)

    return gal_data_domain, gal_rm, gal_stddev

def egal_settings(lerm, e_rm, e_rm_err):
    egal_data_domain = ift.makeDomain(ift.UnstructuredDomain((lerm,)))

    egal_rm = ift.Field(egal_data_domain, e_rm)
    egal_stddev = ift.Field(egal_data_domain, e_rm_err)

    return egal_data_domain, egal_rm, egal_stddev