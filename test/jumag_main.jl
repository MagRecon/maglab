using MicroMagnetic
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
        geo = MicroMagnetic.Cylinder(radius=5e-9)
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
    cubic = add_cubic_anis(sim, 4.5e5)

    init_m0(sim, init_m0_fun)

    num = length(sim.interactions)
    all_energy = zeros(num,nx,ny,nz)
    all_field = zeros(num,3,nx,ny,nz)
    for (i, interaction) in enumerate(sim.interactions)
        MicroMagnetic.effective_field(interaction, sim, sim.spin, 0.0)
        all_energy[i,:,:,:] = reshape(interaction.energy, (nx,ny,nz))
        all_field[i,:,:,:,:] = reshape(interaction.field, (3,nx,ny,nz))
    end
    geo = sim.mu0_Ms / (8e5*MicroMagnetic.mu_0)
    return all_energy, all_field, reshape(sim.spin, (3,nx,ny,nz)), reshape(geo, (nx,ny,nz))
end

E, H, m, ms = gen_energy(pbc="")
npzwrite("dataset/energy.npy", E)
npzwrite("dataset/field.npy", H)
npzwrite("dataset/m0.npy", m)

E, H, m, ms = gen_energy(pbc="", cylinder=true)
npzwrite("dataset/energy_cylinder.npy", E)
npzwrite("dataset/field_cylinder.npy", H)
npzwrite("dataset/m0_cylinder.npy", m)

E, H, m, ms = gen_energy(pbc="xy")
npzwrite("dataset/energy_pbc_xy.npy", E)
npzwrite("dataset/field_pbc_xy.npy", H)