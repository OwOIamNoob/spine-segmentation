class Canal(Component):
    def __init__(self, volume, index):
        super().__init__(volume, index)
        self.pose = None
        # Initialization for pose estimation
        self.pcd_model = o3d.geometry.PointCloud()
        self.pcd_model.points = o3d.utility.Vector3dVector(self.pcd)


    def pose_estimate(  self,
                        down_sample=0.08,
                        max_contraction=128,
                        max_attraction=512,
                        step_wise_contraction_amplification=3,
                        termination_ratio=0.0005,
                        init_attraction=1.,
                        init_contraction=1.,
                        max_iteration_steps=50,
                        filter_nb_neighbors=625,
                        filter_std_ratio=20.,
                        sampling=20):
        lbc = pc_skeletor.LBC(  point_cloud=self.pcd_model,
                                down_sample=down_sample,
                                max_contraction=max_contraction,
                                max_attraction=max_attraction,
                                step_wise_contraction_amplification=step_wise_contraction_amplification,
                                termination_ratio=termination_ratio,
                                init_attraction=init_attraction,
                                init_contraction=init_contraction,
                                max_iteration_steps=max_iteration_steps,
                                filter_nb_neighbors=filter_nb_neighbors,
                                filter_std_ratio=filter_std_ratio)
        
        self.pose = lbc.extract_skeleton()[::sampling].tolist()
        # dunnu if i should sort by topology or axis
        self.pose = sorted(self.pose, key=lambda x: x[0])

        topo = lbc.extract_topology()
        return topo

    
