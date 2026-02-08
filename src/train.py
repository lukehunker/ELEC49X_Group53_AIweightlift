import Bar_Tracking.barspeed_to_excel as BST
import OpenFace.test_openface as OF
import LGBM_Regressor.LGBMTrain011123 as LGBM

if __name__ == "__main__":
    BST.run()
    OF.run()
    LGBM.run()