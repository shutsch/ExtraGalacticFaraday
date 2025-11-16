import nifty8 as ift
import libs as Egf 
import numpy as np



def rm_noise(params, data, dest_data, rm_err, ltheta, lerm, z_indices, figname):

        if params['params_mock_cat.maker_params.surveys.make_survey1'] == True:
          
            cat_index_1=np.where(dest_data['catalog']==params['params_mock_cat.maker_params.surveys.name1'])[0]
            sigma_1 = rm_err[np.where(data['catalog']==params['params_mock_cat.maker_params.surveys.cat1'])[0]]
            sigma_mock=np.empty(dest_data['catalog'].size)
            sigma_mock[cat_index_1]=Egf.sampling_from_noise_distribiution(sigma_1, len(cat_index_1))

            if params['params_mock_cat.maker_params.surveys.make_survey2'] == True:
                cat_index_2=np.where(dest_data['catalog']==params['params_mock_cat.maker_params.surveys.name2'])[0]
                sigma_2 = rm_err[np.where(data['catalog']==params['params_mock_cat.maker_params.surveys.cat2'])[0]]
                sigma_mock[cat_index_2]=Egf.sampling_from_noise_distribiution(sigma_2, len(cat_index_2))
                
            sigma_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(dest_data['catalog'].size),np.array(sigma_mock))
            N = ift.DiagonalOperator(sigma_mock_field**2, domain=ift.UnstructuredDomain(sigma_mock_field.size), sampling_dtype=np.float64)

            print('sigma_mock cat1', sigma_mock[cat_index_1].mean())
            print('sigma_mock cat2', sigma_mock[cat_index_2].mean())

            #rm_data+= N.draw_sample().val
            noise=N.draw_sample().val

            Egf.noise_plot(params, sigma_1, sigma_2, sigma_mock[cat_index_1], sigma_mock[cat_index_2], figname)


            
        else:
            #creating mock sigma gal
            #NVSS cat 2009ApJ...702.1230T
            #LoTSS cat "LoTSS DR2 (O'Sullivan et al. 2022) "
            cat_index_gal=np.where(data['catalog']==params['params_mock_cat.maker_params.cat_gal'])[0][~np.isnan(np.where(data['catalog']==params['params_mock_cat.maker_params.cat_gal'])[0])]
            sigma_gal = data['rm_err'][cat_index_gal]
            sigma_gal_mock=Egf.sampling_from_noise_distribiution(sigma_gal, ltheta-lerm)



            #creating mock sigma eg
            cat_index_eg=np.where(data['catalog']==params['params_mock_cat.maker_params.cat_eg'])[0][~np.isnan(np.where(data['catalog']==params['params_mock_cat.maker_params.cat_eg'])[0])]
            sigma_eg = data['rm_err'][cat_index_eg]
            sigma_eg_mock=Egf.sampling_from_noise_distribiution(sigma_eg,lerm) 



            sigma_mock=np.empty(ltheta)
            sigma_mock[np.isnan(dest_data['z_best'])] = sigma_gal_mock
            sigma_mock[z_indices] = sigma_eg_mock
            sigma_mock_field=ift.Field.from_raw(ift.UnstructuredDomain(ltheta),np.array(sigma_mock))
            N = ift.DiagonalOperator(sigma_mock_field**2, domain=ift.UnstructuredDomain(ltheta), sampling_dtype=np.float64)

            noise= N.draw_sample().val
            print('sigma_mock eg', sigma_eg_mock.mean())
            print('sigma_mock gal', sigma_gal_mock.mean())


            Egf.noise_plot(params, sigma_eg, sigma_gal, sigma_eg_mock, sigma_gal_mock, figname)


        return {'Noise': noise, 'Sigma_noise_mock': sigma_mock}