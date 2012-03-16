function M = centering_matrix(dim);
M = eye(dim)-ones(dim)/dim;