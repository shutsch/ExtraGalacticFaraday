import nifty8 as ift
import libs as Egf 





def base_catalog_extractor(base_catalog):
    
    #data catalogs
    data = base_catalog if base_catalog is not None else \
        Egf.get_rm(filter_pulsars=True, version='custom', default_error_level=0.5)

    
    #reading from base_catalog
    catalogs=Egf.get_data(data)

    #eg data
    z_indices = catalogs['z indices']
    e_z = catalogs['e_z']
    e_F = catalogs['e_F']


    #full data
    rm_err = catalogs['rm_err']   

    return  {'z indices': z_indices, 'e_z': e_z,'e_F': e_F, 'rm_err': rm_err}

    