function Rxx=real_imag_consits(real_imag,diags,M)
    reals=real_imag(1:end/2);
    imags=real_imag(end/2+1:end);
    Rxx=zeros(M,M);
    nn=1;
    nnn=1;
    for ii=1:M
        for jj=ii:M
            if ii==jj
                Rxx(ii,jj)=diags(nnn);
                nnn=nnn+1;
                continue;
            end
            Rxx(ii,jj)=reals(nn)+1j*imags(nn);
            Rxx(jj,ii)=reals(nn)-1j*imags(nn);
            nn=nn+1;
        end

    end

end