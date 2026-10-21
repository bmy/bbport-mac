#ifndef R600_IMAGE_BUFFER_H
#define R600_IMAGE_BUFFER_H

static inline unsigned r600_image_buffer_offset(const struct r600_context *const rctx,
						const bool ssbo,
						const mesa_shader_stage stage)
{
	switch ((unsigned)ssbo) {
	case 0:
		switch (stage) {
		case MESA_SHADER_VERTEX: default: return 0;
		case MESA_SHADER_FRAGMENT: return rctx->vs_shader->current->shader.num_images;
		}
	default:
		switch (stage) {
		case MESA_SHADER_VERTEX: default: return rctx->vs_shader->current->shader.num_images +
				rctx->ps_shader->current->shader.num_images;
		case MESA_SHADER_FRAGMENT: return rctx->vs_shader->current->shader.num_images +
				rctx->ps_shader->current->shader.num_images +
				rctx->vs_shader->current->shader.num_ssbos;
		case MESA_SHADER_TESS_CTRL: return rctx->vs_shader->current->shader.num_images +
				rctx->ps_shader->current->shader.num_images +
				rctx->vs_shader->current->shader.num_ssbos +
				rctx->ps_shader->current->shader.num_ssbos;
		case MESA_SHADER_TESS_EVAL: return rctx->vs_shader->current->shader.num_images +
				rctx->ps_shader->current->shader.num_images +
				rctx->vs_shader->current->shader.num_ssbos +
				rctx->ps_shader->current->shader.num_ssbos +
				(rctx->tcs_shader ? rctx->tcs_shader->current->shader.num_ssbos : 0);
		case MESA_SHADER_GEOMETRY: return rctx->vs_shader->current->shader.num_images +
				rctx->ps_shader->current->shader.num_images +
				rctx->vs_shader->current->shader.num_ssbos +
				rctx->ps_shader->current->shader.num_ssbos +
				(rctx->tcs_shader ? rctx->tcs_shader->current->shader.num_ssbos : 0) +
				(rctx->tes_shader ? rctx->tes_shader->current->shader.num_ssbos : 0);
		}
	}
}

#endif
