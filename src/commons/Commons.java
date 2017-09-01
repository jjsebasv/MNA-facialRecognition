package commons;

/**
 * Created by sebastian on 8/29/17.
 */
public class Commons {

    public Matrix hessenberg(Matrix A) {
        Matrix L = Matrix.zeros(A.getRows(), A.getCols());
        Matrix H = A;
        double n = A.getCols();
        // j counts rows
        for (int j = 0; j < A.getRows()-2; j++) {
            Matrix x = H.subMatrix(j+1,A.getCols()-1,A.getCols()-1,A.getCols()-1);
            x.matrix[0] = x.sum(x.sign().multiply(x.norm())).matrix[0];
            n = x.norm();
            Matrix u = Matrix.zeros(x.getRows(),x.getCols());
            if (n > 0) {
                u = x.divide(n);
                H.subMatrix(j+1,A.getCols()-1,j,A.getCols()) =
                        H.subMatrix(j+1,A.getCols()-1,j,A.getCols()).substract(
                                u.multiply(2)
                        )
            }
        }
    }
}
