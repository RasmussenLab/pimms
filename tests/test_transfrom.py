import numpy.testing as npt
import pandas as pd
import pytest
import sklearn
from sklearn import impute, preprocessing

from pimmslearn.io.datasets import to_tensor
from pimmslearn.transform import VaepPipeline

from numpy import nan


data = {
    'feat_00': {'sample_023': 18.475502941566493,
                'sample_156': 22.535341434252544,
                'sample_088': 7.413097020304393,
                'sample_071': 22.127469047666743,
                'sample_040': 5.068051892164936,
                'sample_160': 10.21812499743875,
                'sample_046': 9.71037709212731,
                'sample_137': 20.07067011987325,
                'sample_180': 15.638881429624169,
                'sample_029': 21.0290739437736},
    'feat_01': {'sample_023': 19.052809526029314,
                'sample_156': nan,
                'sample_088': 4.78634040339565,
                'sample_071': 7.1633143730490705,
                'sample_040': 8.35771017095876,
                'sample_160': nan,
                'sample_046': 12.763093158492504,
                'sample_137': 13.868866830018217,
                'sample_180': 29.280414698376408,
                'sample_029': 2.182890190925806},
    'feat_02': {'sample_023': 1.3591202931613355,
                'sample_156': nan,
                'sample_088': 26.15350699776605,
                'sample_071': 3.3142233939418397,
                'sample_040': 5.3103145283024045,
                'sample_160': 12.28886684685696,
                'sample_046': 15.22831136053365,
                'sample_137': 12.352966247535255,
                'sample_180': 3.766506442879601,
                'sample_029': 24.655801778710686},
    'feat_03': {'sample_023': 11.238378438794136,
                'sample_156': 10.643138755079526,
                'sample_088': 6.57641962074133,
                'sample_071': 10.638664729223295,
                'sample_040': 2.6610760127116686,
                'sample_160': 9.336524851901807,
                'sample_046': 7.2722919724524075,
                'sample_137': 19.529204014528624,
                'sample_180': nan,
                'sample_029': 21.187266814694887},
    'feat_04': {'sample_023': 18.77579747142709,
                'sample_156': 28.25177134856775,
                'sample_088': 29.275957674573938,
                'sample_071': 8.617169749622452,
                'sample_040': 3.6190761330180243,
                'sample_160': 20.313619192050282,
                'sample_046': 3.445104742176106,
                'sample_137': 16.362956082207162,
                'sample_180': 23.10474635271288,
                'sample_029': 2.440463419256993},
    'feat_05': {'sample_023': 15.094087757402631,
                'sample_156': 20.05717940168742,
                'sample_088': 10.106873753133206,
                'sample_071': 8.889243613679703,
                'sample_040': 13.823363040981773,
                'sample_160': 18.173377483580712,
                'sample_046': 18.318601273248976,
                'sample_137': 1.8681931764565685,
                'sample_180': 24.214666120946475,
                'sample_029': 2.5451314225557575},
    'feat_06': {'sample_023': 25.694695235649668,
                'sample_156': 20.360098700176945,
                'sample_088': 5.463537470660978,
                'sample_071': 7.008232531497298,
                'sample_040': 6.190011552173775,
                'sample_160': 10.9378118459262,
                'sample_046': 8.658916597207673,
                'sample_137': 15.375079463109952,
                'sample_180': 3.6062168736172873,
                'sample_029': 29.599187355035262},
    'feat_07': {'sample_023': 19.76080894856835,
                'sample_156': 10.857599725135948,
                'sample_088': 23.690955214274375,
                'sample_071': 1.2627956890908565,
                'sample_040': 10.928095831442263,
                'sample_160': 6.536776627649703,
                'sample_046': 17.437146642678368,
                'sample_137': 24.192108516985186,
                'sample_180': 7.967484988758182,
                'sample_029': 11.228123872683609},
    'feat_08': {'sample_023': 4.888032812442891,
                'sample_156': 17.809823736256153,
                'sample_088': 19.761233265026284,
                'sample_071': 0.5362180420024143,
                'sample_040': 15.102518125645707,
                'sample_160': 29.641078984667757,
                'sample_046': 4.630881458226069,
                'sample_137': 13.777196327908742,
                'sample_180': nan,
                'sample_029': 11.119264412006727},
    'feat_09': {'sample_023': 2.1170624220128955,
                'sample_156': nan,
                'sample_088': 14.945871493594169,
                'sample_071': 29.631671692080946,
                'sample_040': 20.71184485888096,
                'sample_160': 13.620048635883446,
                'sample_046': 14.434203055644524,
                'sample_137': 1.5586973266280524,
                'sample_180': 8.799258323416941,
                'sample_029': 24.383987017725076},
    'feat_10': {'sample_023': 19.27257834618947,
                'sample_156': 19.082881392286268,
                'sample_088': 16.660906528128937,
                'sample_071': 12.83319401207587,
                'sample_040': 1.179364195232968,
                'sample_160': 20.64822707340711,
                'sample_046': 15.977682976547577,
                'sample_137': 23.588345981520995,
                'sample_180': 23.194256217560145,
                'sample_029': 28.417457321515762},
    'feat_11': {'sample_023': 0.7953393162486544,
                'sample_156': 27.3986084272871,
                'sample_088': 21.576053348167914,
                'sample_071': 11.52979941479045,
                'sample_040': 23.98231196727128,
                'sample_160': 4.216589027350823,
                'sample_046': 1.5547061046728072,
                'sample_137': 6.040913464079232,
                'sample_180': 15.538815831377368,
                'sample_029': 29.580031914686128},
    'feat_12': {'sample_023': 17.5732674382039,
                'sample_156': 18.377204050344574,
                'sample_088': 6.853642239938962,
                'sample_071': 20.389418480792095,
                'sample_040': 18.837011684727234,
                'sample_160': 14.567690989781934,
                'sample_046': 10.098128345817617,
                'sample_137': 7.758625046641163,
                'sample_180': 10.442867795048642,
                'sample_029': 22.601345557768248},
    'feat_13': {'sample_023': 28.206907242748727,
                'sample_156': 26.21095780399035,
                'sample_088': 29.89001748170226,
                'sample_071': 6.5476166359519254,
                'sample_040': 2.4527709584661572,
                'sample_160': nan,
                'sample_046': 4.032440308169227,
                'sample_137': 4.941190602968877,
                'sample_180': 11.152178847189909,
                'sample_029': 11.287787565927474},
    'feat_14': {'sample_023': 17.26422533627637,
                'sample_156': 21.719191663982816,
                'sample_088': 29.243794864403494,
                'sample_071': 28.49883551850676,
                'sample_040': 26.207358723203317,
                'sample_160': 15.163653537313499,
                'sample_046': 1.9012491141830323,
                'sample_137': 9.906451946144927,
                'sample_180': nan,
                'sample_029': 2.505021500960063}
}


def test_Vaep_Pipeline():
    dae_default_pipeline = sklearn.pipeline.Pipeline(
        [
            ('normalize', preprocessing.StandardScaler()),
            ('impute', impute.SimpleImputer(add_indicator=False))  # True won't work
        ]
    )
    df = pd.DataFrame(data)
    mask = df.notna()
    # new procs, transform equal encode, inverse_transform equals decode
    dae_transforms = VaepPipeline(df, encode=dae_default_pipeline)
    res = dae_transforms.transform(df)
    assert isinstance(res, pd.DataFrame)
    with pytest.raises(ValueError):
        res = dae_transforms.inverse_transform(res)  # pd.DataFrame
    with pytest.raises(ValueError):
        _ = dae_transforms.inverse_transform(res.iloc[0])  # pd.DataFrame
    with pytest.raises(ValueError):
        _ = dae_transforms.inverse_transform(res.loc['sample_156'])  # pd.DataFrame
    with pytest.raises(ValueError):
        _ = dae_transforms.inverse_transform(to_tensor(res))  # torch.Tensor
    with pytest.raises(ValueError):
        _ = dae_transforms.inverse_transform(res.values)  # numpy.array
    with pytest.raises(ValueError):
        _ = dae_transforms.inverse_transform(res.values[0])  # single sample
    dae_transforms = VaepPipeline(df, encode=dae_default_pipeline, decode=['normalize'])
    res = dae_transforms.transform(df)
    res = dae_transforms.inverse_transform(res)
    npt.assert_array_almost_equal(df.values[mask], res.values[mask])
