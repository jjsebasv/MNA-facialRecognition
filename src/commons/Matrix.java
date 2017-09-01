package commons;

/**
 * Created by sebastian on 8/29/17.
 */
public class Matrix {

    public double [][] matrix;
    private int rows;
    private int cols;

    public Matrix (double[][] m) {
        this.matrix = m;
        this.rows = m.length;
        this.cols = m[0].length;
    }

    public static Matrix zeros (int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        return new Matrix(matrix);
    }

    public Matrix subMatrix (int initialRow, int finalRow, int initialCol, int finalCol) {
        Matrix subMatrix = zeros(finalRow - initialRow + 1, finalCol - initialCol + 1);
        for (int i = 0; i <= rows-finalRow; i ++) {
            for (int j = 0; j <= cols-finalCol; j++) {
                subMatrix.matrix[i][j] = matrix[i+initialRow][j+initialCol];
            }
        }
        return subMatrix;
    }

    public int getRows() {
        return this.rows;
    }

    public int getCols() {
        return this.cols;
    }

    /**
     * Equivalent to infinite norm.
     * @return Maximum absolute row sum
     */
    public double norm() {
        double norm = 0;
        for (int j = 0; j < cols; j++) {
            double aux = 0;
            for (int i = 0; i < rows; i++) {
                aux += Math.abs(matrix[i][j]);
            }
            norm = aux > norm ? aux : norm;
        }
        return norm;
    }

    public Matrix sign() {
        Matrix sign = zeros(rows, cols);
        for (int i = 0; i < rows; i ++) {
            for (int j = 0; j < cols; j++) {
                sign.matrix[i][j] = Math.signum(matrix[i][j]);
            }
        }
        return sign;
    }

    public Matrix multiply(double p) {
        Matrix aux = this;
        for (int i = 0; i < rows; i ++) {
            for (int j = 0; j < cols; j++) {
                aux.matrix[i][j] = p*matrix[i][j];
            }
        }
        return aux;
    }

    public Matrix divide(double p) {
        Matrix aux = this;
        for (int i = 0; i < rows; i ++) {
            for (int j = 0; j < cols; j++) {
                aux.matrix[i][j] = matrix[i][j]/p;
            }
        }
        return aux;
    }

    public Matrix sum(Matrix B) {
        Matrix aux = zeros(rows, cols);
        for (int i = 0; i < rows; i ++) {
            for (int j = 0; j < cols; j++) {
                aux.matrix[i][j] = this.matrix[i][j] + B.matrix[i][j];
            }
        }
        return aux;
    }

    public Matrix substract(Matrix B) {
        Matrix aux = zeros(rows, cols);
        for (int i = 0; i < rows; i ++) {
            for (int j = 0; j < cols; j++) {
                aux.matrix[i][j] = this.matrix[i][j] - B.matrix[i][j];
            }
        }
        return aux;
    }

    public Matrix transpose() {
        Matrix aux = zeros(cols,rows);
        for (int i = 0; i < rows; i ++) {
            for (int j = 0; j < cols; j++) {
                aux.matrix[i][j] = this.matrix[j][i];
            }
        }
        return aux;
    }
}
