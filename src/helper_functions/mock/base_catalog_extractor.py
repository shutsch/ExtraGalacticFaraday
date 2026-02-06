import nifty8 as ift
import libs as Egf 





def base_catalog_extractor(params, base_catalog):
    
    #data catalogs
    #data = base_catalog if base_catalog is not None else \
    #    Egf.get_rm(version=params['file_params.version'], filter_pulsars=True, default_error_level=0.5, params=params, full_catalog_path=None)
    data = Egf.get_rm(version=None, filter_pulsars=True, default_error_level=0.5, params=params, full_catalog_path=params['file_params.cat_path']+base_catalog) if base_catalog is not None else \
        Egf.get_rm(version=params['file_params.version'], filter_pulsars=True, default_error_level=0.5, params=params, full_catalog_path=None)

    return {'base catalog': data}

    