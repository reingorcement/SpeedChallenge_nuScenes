"""Independent world-frame reference checks for the training-only constraint."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from purevision_v5.fusion_geometry import theoretical_flow,geometry_loss,validate_packet
from purevision_v5.fusion_geometry_prepare import resize_intrinsic


def example(velocity=(8.,.7),angle=.03,lever=(1.7,.02,1.51)):
    camera=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    r=Rotation.from_euler('z',angle).as_matrix();l=np.array(lever);dt=.1
    world=np.stack((np.linspace(5.,35.,64),np.linspace(-2.,2.,64),np.zeros(64)),-1)
    source=(world-l)@camera
    # Independent direct composition of world point -> target ego -> target camera.
    position=r@np.r_[velocity,0.]*dt
    target=((world-position)@r-l)@camera
    k=np.array([[100.,0.,63.5],[0.,100.,35.5],[0.,0.,1.]])
    def project(x):
        h=x@k.T;return h[:,:2]/h[:,2:]
    flow=project(target)-project(source)
    g=dict(points=torch.tensor(source[None],dtype=torch.float32),flow=torch.tensor(flow[None],dtype=torch.float32),
        mask=torch.ones(1,64,dtype=torch.bool),K=torch.tensor(k[None],dtype=torch.float32),
        R21=torch.tensor((camera.T@r.T@camera)[None],dtype=torch.float32),
        base=torch.tensor((camera.T@(r.T@l-l))[None],dtype=torch.float32),
        velocity_map=torch.tensor((-dt*camera.T[:,:2])[None],dtype=torch.float32),
        dt=torch.tensor([dt]),valid=torch.tensor([True]))
    return torch.tensor([velocity],dtype=torch.float32),g


@pytest.mark.parametrize('velocity,angle,lever',[
    ((8.,0.),0.,(0.,0.,1.5)),((0.,2.),0.,(0.,0.,1.5)),
    ((0.,-2.),0.,(0.,0.,1.5)),((0.,0.),.06,(0.,0.,0.)),
    ((0.,0.),.06,(1.7,.1,1.5)),((8.,.7),.03,(1.7,.02,1.51)),
])
def test_world_composition(velocity,angle,lever):
    v,g=example(velocity,angle,lever);validate_packet(g)
    flow,z=theoretical_flow(v,g)
    torch.testing.assert_close(flow,g['flow'],atol=1e-5,rtol=1e-5)
    assert (z>0).all()


def test_velocity_gradient_and_stop_gradient_geometry():
    v,g=example();v=(v+torch.tensor([[1.,-.3]])).requires_grad_()
    for key in ('points','flow','K','R21','base','velocity_map','dt'):g[key].requires_grad_()
    loss=geometry_loss(v,g);loss.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0
    assert all(value.grad is None for value in g.values())


def test_all_windows_equal_weight_and_invalid_window_zero():
    v,g=example();v=v+1
    single=geometry_loss(v,g)
    two={k:torch.cat((x,x),0) for k,x in g.items()};two['valid'][1]=False
    torch.testing.assert_close(geometry_loss(v.repeat(2,1),two),single/2)
    two['mask'][0,32:]=False
    expected=torch.nn.functional.smooth_l1_loss(
        (theoretical_flow(v,g)[0][:,:32]-g['flow'][:,:32])/10,
        torch.zeros(1,32,2),beta=.01,reduction='mean')/2
    torch.testing.assert_close(geometry_loss(v.repeat(2,1),two),expected)


def test_perturbations_increase_synthetic_residual():
    v,g=example();base=geometry_loss(v,g)
    assert base<1e-10
    for delta in ([1.,0.],[-1.,0.],[0.,1.],[0.,-1.]):
        assert geometry_loss(v+torch.tensor([delta]),g)>base+1e-6
    for factor in (.95,1.05):
        changed=dict(g,points=g['points']*factor)
        assert geometry_loss(v,changed)>base+1e-7
    changed=dict(g,dt=g['dt']*1.1,velocity_map=g['velocity_map']*1.1)
    assert geometry_loss(v,changed)>base+1e-6


def test_intrinsics_follow_two_half_pixel_resizes():
    k=np.array([[1260.,0.,816.],[0.,1262.,491.],[0.,0.,1.]])
    actual=resize_intrinsic(k,1600,900)
    assert np.isclose(actual[0,0],1260*128/1600)
    assert np.isclose(actual[1,1],1262*72/900)
    assert np.isclose(actual[0,2],(816+.5)*128/1600-.5)
    assert np.isclose(actual[1,2],(491+.5)*72/900-.5)
