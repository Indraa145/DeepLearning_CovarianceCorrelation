import numpy as np

#Indra Imanuel Gunawan - 20195118
#Deep Learning Homework 1 | No. 10 - 13

#Function to calculate the Covariance and Correlation Matrix
def findCovarianceAndCorrelationMatrix(data):
    mean = []
    #Creating the Mean vector
    for i in range(np.size(data,0)):
        mean.append(np.mean(data[i,:]))
    mean_array = np.array([mean])
    mean_array = np.transpose(mean_array)

    cov_matrix = 0
    for i in range(np.size(data,1)):
        data_row = np.reshape(data[:,i],(np.size(data,0),1)) - mean_array
        data_row_t = np.transpose(data_row)
        matmul = np.matmul(data_row, data_row_t)
        cov_matrix += matmul

    cov_matrix = cov_matrix/(np.size(data,1)-1)
    standardized_matrix = np.linalg.inv(np.sqrt(np.diag(np.diag(cov_matrix))))
    corr_matrix = standardized_matrix.dot(cov_matrix).dot(standardized_matrix)

    print("Covariance matrix:")
    print(cov_matrix)
    print("Correlation matrix:")
    print(corr_matrix)
    print("")

#Because I already made the function, all I need to do is just put the data into the function
#Number 10 (Height, Weight, Grade)
data_no10 = np.array(
    [[170,165,174,169,155,172,166,168],
     [60,55,75,67,49,63,58,61],
     [4.1,3.0,2.8,2.9,3.1,3.6,3.7,4.0]]
)
print("Covariance & Correlation Matrix for Data No. 10:")
findCovarianceAndCorrelationMatrix(data_no10)

#Number 11 (S&P 500, ABC Corp.)
data_no11 = np.array(
    [[1692,1978,1884,2151,2519],
     [68,102,110,112,154]]
)
print("Covariance & Correlation Matrix for Data No. 11:")
findCovarianceAndCorrelationMatrix(data_no11)

#Number 12 (Excelsior Corp. Annual Return (percent) & Adirondack Corp. Annual Return (percent))
data_no12 = np.array(
    [[1,-2,3,0,3],
     [3,2,4,6,0]]
)
print("Covariance & Correlation Matrix for Data No. 12:")
findCovarianceAndCorrelationMatrix(data_no12)

#Number 13 (Temperature, Number of Customers)
data_no13 = np.array(
    [[98,87,90,85,95,75],
    [15,12,10,10,16,7]]
)
print("Covariance & Correlation Matrix for Data No. 13:")
findCovarianceAndCorrelationMatrix(data_no13)