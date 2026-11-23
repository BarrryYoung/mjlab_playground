"""Velocity runner whose ONNX metadata also covers explicit-PD actuators.

mjlab's get_base_metadata reads kp/kd from the MuJoCo actuator params
(gainprm[0], -biasprm[2]). That only holds for builtin position actuators. An
IdealPd/DcMotor actuator computes the PD torque in torch and drives a plain
<motor> (gain 1, bias 0), so the export would say kp=1, kd=0. Here the gains of
those joints are taken from the actuator objects instead; every other joint keeps
mjlab's value, so exports of builtin-actuator tasks are unchanged.
"""

import wandb
from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def get_metadata(env, run_name: str) -> dict:
  metadata = get_base_metadata(env, run_name)
  robot = env.scene["robot"]
  names = list(metadata["joint_names"])
  kp = list(metadata["joint_stiffness"])
  kd = list(metadata["joint_damping"])
  for act in robot.actuators:
    if not isinstance(act, IdealPdActuator):
      continue
    for j, name in enumerate(act.target_names):
      i = names.index(name)
      # Nominal gains: tasks may randomize stiffness/damping per env (pd_gains).
      kp[i] = float(act.default_stiffness[0, j])
      kd[i] = float(act.default_damping[0, j])
  metadata["joint_stiffness"] = kp
  metadata["joint_damping"] = kd
  return metadata


class Go2wVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None):
    super(VelocityOnPolicyRunner, self).save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      attach_metadata_to_onnx(str(onnx_path), get_metadata(self.env.unwrapped, run_name))
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
