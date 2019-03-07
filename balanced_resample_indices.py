def balanced_resample_indices( classes, n_per_class=False, verbose=False, down_w_replacement=False, upsample=False):
 '''
 balanced_resample_indices( classes, n_per_class=False, verbose=False, down_w_replacement=False, upsample=False)
 
 Takes the class labels of a set of samples, and returns the indices needed to randomly resample the set
 such that classes have the desired composition.  By default (no optional args) it downsamples all classes 
 to the size of the smallest class. You can enable replacement sampling (down_w_replacement=True). You can
 downsample all larger classes to n_per_sample and just take all samples of smaller classes (upsample=False), 
 or allow upsampling classes that are smaller than n_per_class in size (upsample=True). Verbose=True results 
 in printing information about which classes of what sizes will be resampled to what sized.
 
 inputs: classes is an ordered list/array/pd.Series of class labels, 1 per sample in the dataset;
         other arguments as described above.
 outputs: np.array of sample indices to create the balanced resample, ie. one should use 
         ret = balanced_resample_indices(...)
         resampled_labels = classes[ret]
         resampled_data = data[ret,:] (or whatever depending on the nature of the data)
 '''
    uniq, ucounts = np.unique( classes, return_counts=True)
    if not n_per_class:
        n_per_class = np.min(ucounts)
    upsamp = ucounts < n_per_class
    downsamp = ucounts > n_per_class
    if verbose:
        sts = ['{}(n={})'.format(c,n) for c,n in zip(uniq[upsamp],ucounts[upsamp]) ]
        if upsample:
            print("The following classes are being upsampled to {}: {}".format(n_per_class,', '.join(sts)))
        sts = ['{}(n={})'.format(c,n) for c,n in zip(uniq[downsamp],ucounts[downsamp]) ]
        print("The following classes are being downsampled to {}: {}".format(n_per_class,', '.join(sts)))
    outi = np.array([],dtype=int)
    for cl,n in zip(uniq,ucounts):
        orig_ind = np.where(classes==cl)[0]
        if n < n_per_class:
            try: # will error if n_per_class > n and replace = False
                outi = np.concatenate((outi, np.random.choice(orig_ind,size=n_per_class,replace=upsample)), axis=None)
            except:
                outi = np.concatenate((outi, orig_ind)) # if we aren't allowed to upsample just take everything to balance as well as possible
        else:
            outi = np.concatenate((outi, np.random.choice(orig_ind,size=n_per_class,replace=down_w_replacement)), axis=None)
    return outi
