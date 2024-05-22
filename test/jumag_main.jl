using JuMag
using NPZ
using Printf

N = 11
nx,ny,nz = N, N ,N

function init_m0_fun(i,j,k,dx,dy,dz)
    qx, qy, qz = 1/2, 1/3, 1/4
    phi = qx * i + qy * j + qz * k
    theta = 2/3 * pi
    return sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)
end


function gen_energy(;pbc="", cylinder=false)
    mesh = FDMesh(nx=nx,ny=ny,nz=nz,dx=1e-9,dy=1e-9,dz=1e-9, pbc=pbc)
    sim = Sim(mesh)
    if cylinder
        geo = JuMag.Cylinder(radius=5e-9)
        set_Ms(sim, geo, 8e5 )
    else
        set_Ms(sim, 8e5)
    end
    exch = add_exch(sim, 1e-12)
    dmi = add_dmi(sim, 1e-4)
    anis = add_anis(sim, 1e3, axis=(0.3,0.4,0.5))
    demag = add_demag(sim)
    zeeman = add_zeeman(sim, (0,0,1e3))
    interfacial_dmi = add_dmi(sim, 2e-4, type="interfacial")

    init_m0(sim, init_m0_fun)

    num = length(sim.interactions)
    all_energy = zeros(num,nx,ny,nz)
    for (i, interaction) in enumerate(sim.interactions)
        JuMag.effective_field(interaction, sim, sim.spin, 0.0)
        all_energy[i,:,:,:] = reshape(interaction.energy, (nx,ny,nz))
    end
    geo = sim.mu0_Ms / (8e5*JuMag.mu_0)
    return all_energy, reshape(sim.spin, (3,nx,ny,nz)), reshape(geo, (nx,ny,nz))
end

e, m, ms = gen_energy(pbc="")
npzwrite("dataset/energy.npy", e)
npzwrite("dataset/m0.npy", m)

e, m, ms = gen_energy(pbc="", cylinder=true)
npzwrite("dataset/energy_cylinder.npy", e)
npzwrite("dataset/m0_cylinder.npy", m)

e, m, ms = gen_energy(pbc="xy")
npzwrite("dataset/energy_pbc_xy.npy", e)