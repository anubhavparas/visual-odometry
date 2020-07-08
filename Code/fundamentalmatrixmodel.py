from constants import *



class FundamentalMatrixModel:
    
    def fit(self, data):
        self.fundamental_mat_model = self.calc_fundamental_matrix(data[:, :3], data[:, 3:])
        return self.fundamental_mat_model


    def evaluate(self, data, threshold):
        error_list = []
        data_size = len(data)
        x1 = np.array(data[:, :3])
        x2 = np.array(data[:, 3:])
        F = self.fundamental_mat_model
        F_trans = F.T
        left_img_inliers = []
        right_img_inliers = []
        inliers_count = 0
        for i in range(data_size):
            
            #'''
            numerator = (x2[i].T @ F @ x1[i])**2
            F_x1 = F @ x1[i]
            F_trans_x2 = F_trans @ x2[i]
            denominator = (F_x1[0])**2 + (F_x1[1])**2 + (F_trans_x2[0])**2 + (F_trans_x2[1])**2
            error = numerator / denominator
            #'''
            #error = abs(x2[i].T @ F @ x1[i])
            if error < threshold:
                left_img_inliers.append(x1[i])
                right_img_inliers.append(x2[i])
                inliers_count += 1
            
            #error_list.append(error)

        return np.ascontiguousarray(left_img_inliers), np.ascontiguousarray(right_img_inliers), inliers_count


    def calc_fundamental_matrix(self, point_list1, point_list2):
        point_list1, point_list2, trans_mat1, trans_mat2 = self.rescale_points(point_list1, point_list2)
        
        A = []
        for i in range(N_POINT):
            A.append(np.kron(point_list2[i], point_list1[i]))
        
        U, S, VT = np.linalg.svd(A)
        fundamental_mat_vec = VT.T[:, -1]
        
        fundamental_mat = np.reshape(fundamental_mat_vec, (3,3))

        fund_mat_U, fund_mat_S, fund_mat_VT = np.linalg.svd(fundamental_mat)
        fund_mat_S = np.diag(fund_mat_S)
        
        
        # enforcing rank 2
        fund_mat_S[2,2] = 0

        #re-estimating the matrix with rank=2
        fundamental_mat = fund_mat_U @ fund_mat_S @ fund_mat_VT

        # rescaling the fundamental_mat
        fundamental_mat = trans_mat2.T @ fundamental_mat @ trans_mat1

        # normalizing fundamental_mat
        #fundamental_mat /= fundamental_mat[2,2]

        return fundamental_mat
    

    def rescale_points(self, point_list1, point_list2):
        mean1 = np.mean(point_list1, axis=0)
        mean2 = np.mean(point_list2, axis=0)

        scale1 = self.get_scale(point_list1[:, :2], mean1[:2])
        scale2 = self.get_scale(point_list2[:, :2], mean2[:2])

        trans_mat1 = np.array([
            [scale1, 0, -scale1*mean1[0]],
            [0, scale1, -scale1*mean1[1]],
            [0, 0, 1]
        ])

        trans_mat2 = np.array([
            [scale2, 0, -scale2*mean2[0]],
            [0, scale2, -scale2*mean2[1]],
            [0, 0, 1]
        ])

        point_list1 = (trans_mat1 @ point_list1.T).T
        point_list2 = (trans_mat2 @ point_list2.T).T

        return point_list1, point_list2, trans_mat1, trans_mat2
    

    def get_scale(self, point_list, mean):
        squares = (point_list-mean)**2
        sum_of_sq = np.sum(point_list, axis = 1)
        mean_ss = np.mean(sum_of_sq) # try np.mean(sum_of_sq)*len(sum_of_sq)
        return np.sqrt(2/mean_ss)





