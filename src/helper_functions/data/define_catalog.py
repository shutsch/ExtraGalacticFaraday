import libs as Egf



def define_catalog(params):

    data = Egf.get_rm(filter_pulsars=True, version=params['file_params.version'], default_error_level=0.5, params=params)

    #create mock catalog option
    if(params['params_mock_cat.maker_params.use_mock']):
        if params['params_mock_cat.maker_params.surveys.make_survey1']==True:
            survey_data=Egf.SurveyMaker(params).make_survey()
            catalog_name=Egf.CatalogMaker(params, base_catalog_data=Egf.base_catalog_extractor(data), base_catalog=data, dest_catalog=survey_data).make_catalog()
            Egf.logger.info("CREATED NEW MOCK SURVEY CATALOG")        
            
            
        else:
            catalog_name=Egf.CatalogMaker(params, base_catalog_data=Egf.base_catalog_extractor(data), base_catalog=data, dest_catalog=None).make_catalog()
            Egf.logger.info("CREATED NEW MOCK CATALOG")       

   
        data = Egf.get_rm(filter_pulsars=True, version=None, full_catalog_path=f'{catalog_name}', default_error_level=0.5, params=params)

    return {'Data for inference': data}


