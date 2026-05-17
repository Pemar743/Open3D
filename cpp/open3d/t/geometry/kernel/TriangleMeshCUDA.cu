// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/TriangleMeshImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace trianglemesh {

void ComputeVertexNormalsCUDA(const core::Tensor& triangles,
                              const core::Tensor& triangle_normals,
                              core::Tensor& vertex_normals) {
    const core::Dtype dtype = vertex_normals.GetDtype();
    const int64_t n = triangles.GetLength();
    const core::Tensor triangles_d = triangles.To(core::Int64);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const int64_t* triangle_ptr = triangles_d.GetDataPtr<int64_t>();
        const scalar_t* triangle_normals_ptr =
                triangle_normals.GetDataPtr<scalar_t>();
        scalar_t* vertex_normals_ptr = vertex_normals.GetDataPtr<scalar_t>();
        core::ParallelFor(
                vertex_normals.GetDevice(), n,
                [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int64_t idx = 3 * workload_idx;
                    int64_t triangle_id1 = triangle_ptr[idx];
                    int64_t triangle_id2 = triangle_ptr[idx + 1];
                    int64_t triangle_id3 = triangle_ptr[idx + 2];

                    scalar_t n1 = triangle_normals_ptr[idx];
                    scalar_t n2 = triangle_normals_ptr[idx + 1];
                    scalar_t n3 = triangle_normals_ptr[idx + 2];

                    atomicAdd(&vertex_normals_ptr[3 * triangle_id1], n1);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id1 + 1], n2);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id1 + 2], n3);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id2], n1);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id2 + 1], n2);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id2 + 2], n3);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id3], n1);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id3 + 1], n2);
                    atomicAdd(&vertex_normals_ptr[3 * triangle_id3 + 2], n3);
                });
    });
}

void ComputeVertexAreasCUDA(const core::Tensor& vertices,
                           const core::Tensor& triangles,
                           core::Tensor& vertex_areas) {
    const int64_t n = triangles.GetLength();
    const core::Dtype dtype = vertex_areas.GetDtype();
    const core::Tensor triangles_d = triangles.To(core::Int64);

    vertex_areas.Fill(0);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t* area_ptr = vertex_areas.GetDataPtr<scalar_t>();
        const int64_t* triangle_ptr = triangles_d.GetDataPtr<int64_t>();
        const scalar_t* vertex_ptr = vertices.GetDataPtr<scalar_t>();

        core::ParallelFor(vertex_areas.GetDevice(), n,
                [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int64_t idx = 3 * workload_idx;

                    int64_t i0 = triangle_ptr[idx];
                    int64_t i1 = triangle_ptr[idx + 1];
                    int64_t i2 = triangle_ptr[idx + 2];

                    scalar_t v01[3], v02[3];

                    v01[0] = vertex_ptr[3 * i1] - vertex_ptr[3 * i0];
                    v01[1] = vertex_ptr[3 * i1 + 1] - vertex_ptr[3 * i0 + 1];
                    v01[2] = vertex_ptr[3 * i1 + 2] - vertex_ptr[3 * i0 + 2];

                    v02[0] = vertex_ptr[3 * i2] - vertex_ptr[3 * i0];
                    v02[1] = vertex_ptr[3 * i2 + 1] - vertex_ptr[3 * i0 + 1];
                    v02[2] = vertex_ptr[3 * i2 + 2] - vertex_ptr[3 * i0 + 2];

                    scalar_t tri_area = scalar_t(0.5) *
                                        core::linalg::kernel::cross_mag_3x1(v01, v02);

                    scalar_t share = tri_area / scalar_t(3.0);

                    atomicAdd(&area_ptr[i0], share);
                    atomicAdd(&area_ptr[i1], share);
                    atomicAdd(&area_ptr[i2], share);
                });
    });
}

}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
