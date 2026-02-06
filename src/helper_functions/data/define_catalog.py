import libs as Egf



def define_catalog(params, base_catalog=None):

    print("Used base catalog:", base_catalog)
    ##data = Egf.get_rm(filter_pulsars=True, version=params['file_params.version'], default_error_level=0.5, params=params)
    data= Egf.base_catalog_extractor(params,base_catalog)['base catalog'] 

    #create mock catalog option
    if(params['params_mock_cat.maker_params.use_mock']):
        if params['params_mock_cat.maker_params.surveys.make_survey1']==True:
            survey_data=Egf.SurveyMaker(params).make_survey()
            catalog_name=Egf.CatalogMaker(params, base_catalog=data, dest_catalog=survey_data).make_catalog()
            Egf.logger.info("CREATED NEW MOCK SURVEY CATALOG")        
            
            
        else:
            catalog_name=Egf.CatalogMaker(params, base_catalog=data, dest_catalog=None).make_catalog()
            Egf.logger.info("CREATED NEW MOCK CATALOG")       

   
        data = Egf.get_rm(version=None, filter_pulsars=True, default_error_level=0.5,  params=params, full_catalog_path=f'{catalog_name}')

    return {'Data for inference': data, 'Catalog path': catalog_name}


