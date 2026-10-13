"""Subtasks of pack tools into toolboxes."""
from core.grade_episode import Subtask


def subtasks():
    return [
        Subtask('pick the hammer for box 1 and seat it at its pose in the toolbox',
                lambda s: ((s.pose_within(obj='hammer0', rel_to='toolbox0', pos=[0.02105573495159139, 0.00967632812402739, -0.012205935335266848], quat=[0.2952809078552302, 0.30677086204324455, 0.6312317851639321, 0.6482708183746906], pos_tol=0.012, rot_tol_deg=15)
                    and s.in_bbox(a='hammer0', b='toolbox0'))
                and s.robot_clear(bodies=['hammer0']))),
        Subtask('pick the pliers for box 1 and seat it at its pose in the toolbox',
                lambda s: ((s.pose_within(obj='pliers0', rel_to='toolbox0', pos=[-0.017805165923998712, -0.05270414678889236, -0.009516564821762866], quat=[0.6339431786891917, -0.6422288781782778, 0.29298933501728497, -0.3159335103996464], pos_tol=0.01, rot_tol_deg=15)
                    and s.in_bbox(a='pliers0', b='toolbox0'))
                and s.robot_clear(bodies=['pliers0']))),
        Subtask('pick the tape measure for box 1 and seat it at its pose in the toolbox',
                lambda s: ((s.pose_within(obj='tape_measure0', rel_to='toolbox0', pos=[-0.08856416604326456, -0.0652194101616544, -0.006150746414827945], quat=[-0.6772636874471636, 0.703957881822565, 0.16502322994017407, -0.13609460593818792], pos_tol=0.01, rot_tol_deg=18)
                    and s.in_bbox(a='tape_measure0', b='toolbox0'))
                and s.robot_clear(bodies=['tape_measure0']))),
        Subtask('pick the wrench for box 1 and seat it at its pose in the toolbox',
                lambda s: ((s.pose_within(obj='wrench0', rel_to='toolbox0', pos=[0.0743051507223057, 0.04840737771880153, -0.010451178323168952], quat=[0.35412027374936855, -0.3224756861633467, -0.6229368778885054, 0.6185195736754595], pos_tol=0.01, rot_tol_deg=15)
                    and s.in_bbox(a='wrench0', b='toolbox0'))
                and s.robot_clear(bodies=['wrench0']))),
        Subtask('pick the hammer for box 2 and seat it at its pose in the toolbox',
                lambda s: ((s.pose_within(obj='hammer1', rel_to='toolbox1', pos=[0.02105573495159139, 0.00967632812402739, -0.012205935335266848], quat=[0.2952809078552302, 0.30677086204324455, 0.6312317851639321, 0.6482708183746906], pos_tol=0.012, rot_tol_deg=15)
                    and s.in_bbox(a='hammer1', b='toolbox1'))
                and s.robot_clear(bodies=['hammer1']))),
        Subtask('pick the pliers for box 2 and seat it at its pose in the toolbox',
                lambda s: ((s.pose_within(obj='pliers1', rel_to='toolbox1', pos=[-0.017805165923998712, -0.05270414678889236, -0.009516564821762866], quat=[0.6339431786891917, -0.6422288781782778, 0.29298933501728497, -0.3159335103996464], pos_tol=0.01, rot_tol_deg=15)
                    and s.in_bbox(a='pliers1', b='toolbox1'))
                and s.robot_clear(bodies=['pliers1']))),
        Subtask('pick the tape measure for box 2 and seat it at its pose in the toolbox',
                lambda s: ((s.pose_within(obj='tape_measure1', rel_to='toolbox1', pos=[-0.08856416604326456, -0.0652194101616544, -0.006150746414827945], quat=[-0.6772636874471636, 0.703957881822565, 0.16502322994017407, -0.13609460593818792], pos_tol=0.01, rot_tol_deg=18)
                    and s.in_bbox(a='tape_measure1', b='toolbox1'))
                and s.robot_clear(bodies=['tape_measure1']))),
        Subtask('pick the wrench for box 2 and seat it at its pose in the toolbox',
                lambda s: s.success()),
    ]
