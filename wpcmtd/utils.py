import datetime

def adjust_date_range(start_date, end_date, init_inc):
    """
    Adjust beginning and end dates to proper ranges given initialization increment 
    (init_inc, in hours; e.g. total hours between model init. cycles)

    Parameters
    ----------
    start_date : datetime.datetime object
        Start date of model data to be loaded.
    end_date : datetime.datetime object
        End date of model data to be loaded.
    
    Returns
    -------
    start_date : datetime.datetime object
        Adjusted start date.
    end_date : datetime.datetime object
        Adjusted end date.
    """
    start_date = start_date + datetime.timedelta(hours = (init_inc-start_date.hour%init_inc)*(start_date.hour%init_inc>0))
    end_date = end_date + datetime.timedelta(hours = init_inc) - datetime.timedelta(hours = end_date.hour%init_inc)

    #Throw an error if 'end_date' is before 'start_date'
    if (end_date - start_date).days < 0:
        raise ValueError("'end_date' is too close to 'start_date'")

    return (start_date, end_date)




def setup_model_data_mtd(grib_path, load_models, temp_dir=None,):
    print("HERE") 