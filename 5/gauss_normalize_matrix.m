function [out,para]=gauss_normalize_matrix(matrix,para_in)
if isempty(para_in)
    para.std=std(matrix,0,1);
    para.mean=mean(matrix,1);
    out=(matrix-ones(size(matrix))*diag(para.mean))./(ones(size(matrix))*diag(para.std));
else
    out=(matrix-ones(size(matrix))*diag(para_in.mean))./(ones(size(matrix))*diag(para_in.std));
    para=para_in;
end
end