#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <functional>
#include <algorithm>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char*> validation_layers =
{
	"VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> device_extensions =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* create_info, const VkAllocationCallbacks* allocator, VkDebugReportCallbackEXT* callback)
{
	auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (func)
		return func(instance, create_info, allocator, callback);
	else
		return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* allocator)
{
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func)
		func(instance, callback, allocator);
}

static std::vector<char> read_file(const std::string& file_name)
{
	std::ifstream file(file_name, std::ios::ate | std::ios::binary);

	if (!file.is_open())
		throw std::runtime_error("Failed to open file");

	size_t file_size = (size_t)file.tellg();

	std::vector<char> buffer(file_size);

	file.seekg(0);
	file.read(buffer.data(), file_size);

	file.close();

	return buffer;
}

struct QueueFamilyIndices
{
	int graphics_family = -1;
	int present_family = 1;

	bool is_complete(void)
	{
		return graphics_family >= 0 && present_family >= 0;
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 colour;

	static VkVertexInputBindingDescription get_binding_desc(void)
	{
		VkVertexInputBindingDescription binding_desc = {};
		binding_desc.binding = 0;
		binding_desc.stride = sizeof(Vertex);
		binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return binding_desc;
	}

	static std::array<VkVertexInputAttributeDescription, 2> get_attribute_desc(void)
	{
		std::array<VkVertexInputAttributeDescription, 2> attribute_descs = {};
		attribute_descs[0].binding = 0;
		attribute_descs[0].location = 0;
		attribute_descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descs[0].offset = offsetof(Vertex, pos);

		attribute_descs[1].binding = 0;
		attribute_descs[1].location = 1;
		attribute_descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descs[1].offset = offsetof(Vertex, colour);

		return attribute_descs;
	}
};

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
	{{ -0.5f, -0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f }},
	{{  0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f }},
	{{  0.5f,  0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }},
	{{ -0.5f,  0.5f, 0.0f }, { 1.0f, 1.0f, 1.0f }},
	{{ 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f}}
};

const std::vector<uint16_t> indices = {
	0, 2, 1, 
	2, 0, 3,

	2, 3, 4,
	1, 2, 4,
	3, 0, 4,
	0, 1, 4
};

class hello_triangle_application
{
public:

	void run(void)
	{
		init_window();
		init_vulkan();
		main_loop();
		cleanup();
	}

private:

	GLFWwindow* _window;
	VkInstance _instance;
	VkPhysicalDevice _physical_device;
	VkDevice _device;
	VkDebugReportCallbackEXT _callback;
	VkQueue _gfx_queue;
	VkQueue _present_queue;
	VkSurfaceKHR _surface;
	VkSwapchainKHR _swap_chain;
	VkRenderPass _render_pass;
	VkDescriptorSetLayout _descriptor_set_layout;
	VkDescriptorPool _descriptor_pool;
	VkDescriptorSet _descriptor_set;
	VkPipelineLayout _pipeline_layout;
	VkPipeline _pipeline;
	VkCommandPool _command_pool;

	VkBuffer _vertex_buffer;
	VkDeviceMemory _vertex_buffer_memory;
	VkBuffer _index_buffer;
	VkDeviceMemory _index_buffer_memory;
	VkBuffer _uniform_buffer;
	VkDeviceMemory _uniform_buffer_memory;

	QueueFamilyIndices _queue_family_indices;

	std::vector<VkImage> _swap_chain_images;
	VkFormat _swap_chain_image_format;
	VkExtent2D _swap_chain_extent;

	std::vector<VkImageView> _swap_chain_image_views;
	std::vector<VkFramebuffer> _swap_chain_framebuffers;
	std::vector<VkCommandBuffer> _command_buffers;

	VkSemaphore _image_available_semaphore;
	VkSemaphore _render_finished_semaphore;

	void init_window(void)
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

		glfwSetWindowUserPointer(_window, this);
		glfwSetWindowSizeCallback(_window, hello_triangle_application::on_window_resize);
	}

	void init_vulkan(void)
	{
		create_instance();
		setup_debug_callback();
		create_surface();
		pick_physical_device();
		create_logical_device();
		create_swap_chain();
		create_image_views();
		create_render_pass();
		create_descriptor_set_layout();
		create_graphics_pipeline();
		create_framebuffers();
		create_command_pool();
		create_vertex_buffer();
		create_index_buffer();
		create_uniform_buffer();
		create_descriptor_pool();
		create_descriptor_set();
		create_command_buffers();
		create_semaphores();
	}

	void main_loop(void)
	{
		while (!glfwWindowShouldClose(_window))
		{
			glfwPollEvents();

			update_uniform_buffer();
			draw_frame();
		}

		vkDeviceWaitIdle(_device);
	}

	void update_uniform_buffer(void)
	{
		static auto start = std::chrono::high_resolution_clock::now();

		auto current = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(current - start).count();

		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), _swap_chain_extent.width / (float)_swap_chain_extent.height, 0.1f, 10.0f);

		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(_device, _uniform_buffer_memory, 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(_device, _uniform_buffer_memory);
	}

	void draw_frame(void)
	{
		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(_device, _swap_chain, std::numeric_limits<uint64_t>::max(), _image_available_semaphore, VK_NULL_HANDLE, &image_index);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreate_swapchain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("Failed to acquire swap chain image");
		}

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore wait_semaphores[] = { _image_available_semaphore };
		VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = wait_semaphores;
		submit_info.pWaitDstStageMask = wait_stages;

		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &_command_buffers[image_index];

		VkSemaphore signal_semaphores[] = { _render_finished_semaphore };
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signal_semaphores;

		if (vkQueueSubmit(_gfx_queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS)
			throw std::runtime_error("Failed to submit draw command buffer");

		VkPresentInfoKHR present_info = {};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;

		VkSwapchainKHR swap_chains[] = { _swap_chain };
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swap_chains;
		present_info.pImageIndices = &image_index;

		result = vkQueuePresentKHR(_present_queue, &present_info);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
		{
			recreate_swapchain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to present swap chain image");
		}

		vkQueueWaitIdle(_present_queue);
	}

	void cleanup(void)
	{
		cleanup_swapchain();

		vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);

		vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, nullptr);

		vkDestroyBuffer(_device, _uniform_buffer, nullptr);
		vkFreeMemory(_device, _uniform_buffer_memory, nullptr);

		vkDestroyBuffer(_device, _index_buffer, nullptr);
		vkFreeMemory(_device, _index_buffer_memory, nullptr);

		vkDestroyBuffer(_device, _vertex_buffer, nullptr);
		vkFreeMemory(_device, _vertex_buffer_memory, nullptr);

		vkDestroySemaphore(_device, _render_finished_semaphore, nullptr);
		vkDestroySemaphore(_device, _image_available_semaphore, nullptr);

		vkDestroyCommandPool(_device, _command_pool, nullptr);

		vkDestroyDevice(_device, nullptr);

		DestroyDebugReportCallbackEXT(_instance, _callback, nullptr);

		vkDestroySurfaceKHR(_instance, _surface, nullptr);
		vkDestroyInstance(_instance, nullptr);

		glfwDestroyWindow(_window);
		glfwTerminate();
	}

	void recreate_swapchain(void)
	{
		int width, height;
		glfwGetWindowSize(_window, &width, &height);

		if (width == 0 || height == 0)
			return;

		vkDeviceWaitIdle(_device);

		cleanup_swapchain();

		create_swap_chain();
		create_image_views();
		create_render_pass();
		create_graphics_pipeline();
		create_framebuffers();
		create_command_buffers();
	}

	void cleanup_swapchain(void)
	{
		for (size_t i = 0; i < _swap_chain_framebuffers.size(); i++)
			vkDestroyFramebuffer(_device, _swap_chain_framebuffers[i], nullptr);

		vkFreeCommandBuffers(_device, _command_pool, static_cast<uint32_t>(_command_buffers.size()), _command_buffers.data());

		vkDestroyPipeline(_device, _pipeline, nullptr);
		vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
		vkDestroyRenderPass(_device, _render_pass, nullptr);

		for (size_t i = 0; i < _swap_chain_image_views.size(); i++)
			vkDestroyImageView(_device, _swap_chain_image_views[i], nullptr);

		vkDestroySwapchainKHR(_device, _swap_chain, nullptr);
	}

	void create_instance(void)
	{
		VkApplicationInfo app_info = {};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName = "Hello Triangle";
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName = "No Engine";
		app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_0;

		auto extensions = get_required_extensions();

		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pApplicationInfo = &app_info;
		create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		create_info.ppEnabledExtensionNames = extensions.data();

		if (enable_validation_layers)
		{
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
			create_info.enabledLayerCount = 0;
		}

		if (vkCreateInstance(&create_info, nullptr, &_instance) != VK_SUCCESS)
			throw std::runtime_error("Failed to create instance");
	}

	void create_surface()
	{
		if (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface) != VK_SUCCESS)
			throw std::runtime_error("Failed to create window surface");
	}

	void pick_physical_device(void)
	{
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(_instance, &device_count, nullptr);

		if (device_count == 0)
			throw std::runtime_error("Failed to find GPUs with Vulkan support");

		std::vector<VkPhysicalDevice> devices(device_count);
		vkEnumeratePhysicalDevices(_instance, &device_count, devices.data());

		for (const auto& device : devices)
		{
			if (is_device_suitable(device))
			{
				_physical_device = device;
				break;
			}
		}

		if (_physical_device == VK_NULL_HANDLE)
			throw std::runtime_error("Failed to find a suitable GPU");
	}

	void create_logical_device(void)
	{
		std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
		std::set<int> unique_queue_families = { _queue_family_indices.graphics_family, _queue_family_indices.present_family };

		float queue_priority = 0.0f;
		for (int queue_family : unique_queue_families)
		{
			VkDeviceQueueCreateInfo queue_create_info = {};
			queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_create_info.queueFamilyIndex = queue_family;
			queue_create_info.queueCount = 1;
			queue_create_info.pQueuePriorities = &queue_priority;
			queue_create_infos.push_back(queue_create_info);
		}

		VkPhysicalDeviceFeatures device_features = {};

		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.pQueueCreateInfos = queue_create_infos.data();
		create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
		create_info.pEnabledFeatures = &device_features;
		create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
		create_info.ppEnabledExtensionNames = device_extensions.data();

		if (enable_validation_layers)
		{
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
			create_info.enabledLayerCount = 0;
		}

		if (vkCreateDevice(_physical_device, &create_info, nullptr, &_device) != VK_SUCCESS)
			throw std::runtime_error("Failed to create logical devie");

		vkGetDeviceQueue(_device, _queue_family_indices.graphics_family, 0, &_gfx_queue);
		vkGetDeviceQueue(_device, _queue_family_indices.present_family, 0, &_present_queue);
	}

	void create_swap_chain(void)
	{
		SwapChainSupportDetails swap_chain_support = query_swap_chain_support(_physical_device);

		VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swap_chain_support.formats);
		VkPresentModeKHR present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
		_swap_chain_extent = choose_swap_extent(swap_chain_support.capabilities);
		_swap_chain_image_format = surface_format.format;

		uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
		if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount)
			image_count = swap_chain_support.capabilities.maxImageCount;

		VkSwapchainCreateInfoKHR create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = _surface;
		create_info.minImageCount = image_count;
		create_info.imageFormat = surface_format.format;
		create_info.imageColorSpace = surface_format.colorSpace;
		create_info.imageExtent = _swap_chain_extent;
		create_info.imageArrayLayers = 1;
		create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		uint32_t queue_family_indices[] = { (uint32_t)_queue_family_indices.graphics_family, (uint32_t)_queue_family_indices.present_family };

		if (_queue_family_indices.graphics_family != _queue_family_indices.present_family)
		{
			create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			create_info.queueFamilyIndexCount = 2;
			create_info.pQueueFamilyIndices = queue_family_indices;
		}
		else
		{
			create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			create_info.queueFamilyIndexCount = 0;
			create_info.pQueueFamilyIndices = nullptr;
		}

		create_info.preTransform = swap_chain_support.capabilities.currentTransform;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		create_info.presentMode = present_mode;
		create_info.clipped = VK_TRUE;
		create_info.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(_device, &create_info, nullptr, &_swap_chain) != VK_SUCCESS)
			throw std::runtime_error("Failed to create swap chain");

		vkGetSwapchainImagesKHR(_device, _swap_chain, &image_count, nullptr);
		_swap_chain_images.resize(image_count);
		vkGetSwapchainImagesKHR(_device, _swap_chain, &image_count, _swap_chain_images.data());
	}

	void create_image_views(void)
	{
		_swap_chain_image_views.resize(_swap_chain_images.size());

		for (size_t i = 0; i < _swap_chain_images.size(); i++)
		{
			VkImageViewCreateInfo create_info = {};
			create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			create_info.image = _swap_chain_images[i];
			create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			create_info.format = _swap_chain_image_format;

			create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			create_info.subresourceRange.baseMipLevel = 0;
			create_info.subresourceRange.levelCount = 1;
			create_info.subresourceRange.baseArrayLayer = 0;
			create_info.subresourceRange.layerCount = 1;

			if (vkCreateImageView(_device, &create_info, nullptr, &_swap_chain_image_views[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to create image views");
		}
	}

	void create_render_pass(void)
	{
		VkAttachmentDescription colour_attachment = {};
		colour_attachment.format = _swap_chain_image_format;
		colour_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colour_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colour_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colour_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colour_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colour_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colour_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colour_attachment_ref = {};
		colour_attachment_ref.attachment = 0;
		colour_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colour_attachment_ref;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 1;
		render_pass_info.pAttachments = &colour_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		render_pass_info.dependencyCount = 1;
		render_pass_info.pDependencies = &dependency;

		if (vkCreateRenderPass(_device, &render_pass_info, nullptr, &_render_pass) != VK_SUCCESS)
			throw std::runtime_error("Failed to create render pass");
	}

	void create_descriptor_set_layout(void)
	{
		VkDescriptorSetLayoutBinding ubo_layout_binding = {};
		ubo_layout_binding.binding = 0;
		ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		ubo_layout_binding.descriptorCount = 1;

		ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layout_info = {};
		layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layout_info.bindingCount = 1;
		layout_info.pBindings = &ubo_layout_binding;

		if (vkCreateDescriptorSetLayout(_device, &layout_info, nullptr, &_descriptor_set_layout) != VK_SUCCESS)
			throw std::runtime_error("Failed to create descriptor set layout");
	}

	void create_graphics_pipeline(void)
	{
		auto vshader_code = read_file("shaders/vert.spv");
		auto fshader_code = read_file("shaders/frag.spv");

		VkShaderModule vert_shader_module = create_shader_module(vshader_code);
		VkShaderModule frag_shader_module = create_shader_module(fshader_code);

		VkPipelineShaderStageCreateInfo vert_shader_stage_info = {};
		vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vert_shader_stage_info.module = vert_shader_module;
		vert_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo frag_shader_stage_info = {};
		frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_shader_stage_info.module = frag_shader_module;
		frag_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stage[] = {
			vert_shader_stage_info, frag_shader_stage_info
		};

		auto bind_desc = Vertex::get_binding_desc();
		auto attrib_desc = Vertex::get_attribute_desc();

		VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.vertexBindingDescriptionCount = 1;
		vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrib_desc.size());
		vertex_input_info.pVertexBindingDescriptions = &bind_desc;
		vertex_input_info.pVertexAttributeDescriptions = attrib_desc.data();

		VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)_swap_chain_extent.width;
		viewport.height = (float)_swap_chain_extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = _swap_chain_extent;

		VkPipelineViewportStateCreateInfo viewport_state = {};
		viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state.viewportCount = 1;
		viewport_state.pViewports = &viewport;
		viewport_state.scissorCount = 1;
		viewport_state.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasteriser = {};
		rasteriser.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasteriser.depthClampEnable = VK_FALSE;
		rasteriser.rasterizerDiscardEnable = VK_FALSE;
		rasteriser.polygonMode = VK_POLYGON_MODE_FILL;
		rasteriser.lineWidth = 1.0f;
		rasteriser.cullMode = VK_CULL_MODE_BACK_BIT;
		rasteriser.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasteriser.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colour_blend_attachment = {};
		colour_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colour_blend_attachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colour_blending = {};
		colour_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colour_blending.logicOpEnable = VK_FALSE;
		colour_blending.logicOp = VK_LOGIC_OP_COPY;
		colour_blending.attachmentCount = 1;
		colour_blending.pAttachments = &colour_blend_attachment;
		colour_blending.blendConstants[0] = 0.0f;
		colour_blending.blendConstants[1] = 0.0f;
		colour_blending.blendConstants[2] = 0.0f;
		colour_blending.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo pipeline_layout_info = {};
		pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_info.setLayoutCount = 1;
		pipeline_layout_info.pSetLayouts = &_descriptor_set_layout;
		pipeline_layout_info.pushConstantRangeCount = 0;

		if (vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_pipeline_layout) != VK_SUCCESS)
			throw std::runtime_error("Failed to create pipeline layout");


		VkGraphicsPipelineCreateInfo pipeline_info = {};
		pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stage;

		pipeline_info.pVertexInputState = &vertex_input_info;
		pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &viewport_state;
		pipeline_info.pRasterizationState = &rasteriser;
		pipeline_info.pMultisampleState = &multisampling;
		pipeline_info.pColorBlendState = &colour_blending;
		pipeline_info.layout = _pipeline_layout;
		pipeline_info.renderPass = _render_pass;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_pipeline) != VK_SUCCESS)
			throw std::runtime_error("Failed to create graphics pipeline");

		vkDestroyShaderModule(_device, vert_shader_module, nullptr);
		vkDestroyShaderModule(_device, frag_shader_module, nullptr);
	}

	void create_framebuffers(void)
	{
		_swap_chain_framebuffers.resize(_swap_chain_image_views.size());

		for (size_t i = 0; i < _swap_chain_image_views.size(); i++)
		{
			VkImageView attachments[] = {
				_swap_chain_image_views[i]
			};

			VkFramebufferCreateInfo framebuffer_info = {};
			framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebuffer_info.renderPass = _render_pass;
			framebuffer_info.attachmentCount = 1;
			framebuffer_info.pAttachments = attachments;
			framebuffer_info.width = _swap_chain_extent.width;
			framebuffer_info.height = _swap_chain_extent.height;
			framebuffer_info.layers = 1;

			if (vkCreateFramebuffer(_device, &framebuffer_info, nullptr, &_swap_chain_framebuffers[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to create framebuffer");
		}
	}

	void create_command_pool(void)
	{
		VkCommandPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = _queue_family_indices.graphics_family;
		pool_info.flags = 0;

		if (vkCreateCommandPool(_device, &pool_info, nullptr, &_command_pool) != VK_SUCCESS)
			throw std::runtime_error("Failed to create command pool");
	}

	void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory)
	{
		VkBufferCreateInfo buffer_info = {};
		buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_info.size = size;
		buffer_info.usage = usage;
		buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(_device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
			throw std::runtime_error("Failed to create vertex buffer");

		VkMemoryRequirements mem_reqs;
		vkGetBufferMemoryRequirements(_device, buffer, &mem_reqs);

		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.size;
		alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, properties);

		if (vkAllocateMemory(_device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate vertex buffer memory");

		vkBindBufferMemory(_device, buffer, buffer_memory, 0);
	}

	void create_vertex_buffer(void)
	{
		VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;

		create_buffer(buffer_size, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

		void* data;
		vkMapMemory(_device, staging_buffer_memory, 0, buffer_size, 0, &data);
		memcpy(data, vertices.data(), (size_t)buffer_size);
		vkUnmapMemory(_device, staging_buffer_memory);

		create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vertex_buffer, _vertex_buffer_memory);

		copy_buffer(staging_buffer, _vertex_buffer, buffer_size);

		vkDestroyBuffer(_device, staging_buffer, nullptr);
		vkFreeMemory(_device, staging_buffer_memory, nullptr);
	}

	void create_index_buffer(void)
	{
		VkDeviceSize buffer_size = sizeof(indices[0]) * indices.size();

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;

		create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

		void* data;

		vkMapMemory(_device, staging_buffer_memory, 0, buffer_size, 0, &data);
		memcpy(data, indices.data(), (size_t) buffer_size);
		vkUnmapMemory(_device, staging_buffer_memory);

		create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _index_buffer, _index_buffer_memory);

		copy_buffer(staging_buffer, _index_buffer, buffer_size);

		vkDestroyBuffer(_device, staging_buffer, nullptr);
		vkFreeMemory(_device, staging_buffer_memory, nullptr);
	}

	void create_uniform_buffer(void)
	{
		VkDeviceSize buffer_size = sizeof(UniformBufferObject);
		create_buffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _uniform_buffer, _uniform_buffer_memory);
	}

	void create_descriptor_pool(void)
	{
		VkDescriptorPoolSize pool_size = {};
		pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		pool_size.descriptorCount = 1;

		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.poolSizeCount = 1;
		pool_info.pPoolSizes = &pool_size;
		pool_info.maxSets = 1;

		if (vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptor_pool) != VK_SUCCESS)
			throw std::runtime_error("Failed to create descriptor pool");
	}

	void create_descriptor_set(void)
	{
		VkDescriptorSetLayout layouts[] = { _descriptor_set_layout };
		VkDescriptorSetAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc_info.descriptorPool = _descriptor_pool;
		alloc_info.descriptorSetCount = 1;
		alloc_info.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(_device, &alloc_info, &_descriptor_set) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate descriptor set");

		VkDescriptorBufferInfo buffer_info = {};
		buffer_info.buffer = _uniform_buffer;
		buffer_info.offset = 0;
		buffer_info.range = sizeof(UniformBufferObject);

		VkWriteDescriptorSet descriptor_write = {};
		descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptor_write.dstSet = _descriptor_set;
		descriptor_write.dstBinding = 0;
		descriptor_write.dstArrayElement = 0;
		descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptor_write.descriptorCount = 1;
		descriptor_write.pBufferInfo = &buffer_info;

		vkUpdateDescriptorSets(_device, 1, &descriptor_write, 0, nullptr);
	}

	void create_command_buffers(void)
	{
		_command_buffers.resize(_swap_chain_framebuffers.size());

		VkCommandBufferAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.commandPool = _command_pool;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandBufferCount = (uint32_t)_command_buffers.size();

		if (vkAllocateCommandBuffers(_device, &alloc_info, _command_buffers.data()) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate command buffers");

		for (size_t i = 0; i < _command_buffers.size(); i++)
		{
			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			begin_info.pInheritanceInfo = nullptr;

			vkBeginCommandBuffer(_command_buffers[i], &begin_info);

			VkRenderPassBeginInfo render_pass_info = {};
			render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			render_pass_info.renderPass = _render_pass;
			render_pass_info.framebuffer = _swap_chain_framebuffers[i];

			render_pass_info.renderArea.offset = { 0, 0 };
			render_pass_info.renderArea.extent = _swap_chain_extent;

			VkClearValue clear_colour = { 0.0f, 0.0f, 0.0f, 1.0f };
			render_pass_info.clearValueCount = 1;
			render_pass_info.pClearValues = &clear_colour;

			vkCmdBeginRenderPass(_command_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);

			VkBuffer vertex_buffers[] = { _vertex_buffer };
			VkDeviceSize offsets[] = { 0 };

			vkCmdBindVertexBuffers(_command_buffers[i], 0, 1, vertex_buffers, offsets);
			vkCmdBindIndexBuffer(_command_buffers[i], _index_buffer, 0, VK_INDEX_TYPE_UINT16);
			vkCmdBindDescriptorSets(_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline_layout, 0, 1, &_descriptor_set, 0, nullptr);

			/*vkCmdDraw(_command_buffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);*/

			vkCmdDrawIndexed(_command_buffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

			vkCmdEndRenderPass(_command_buffers[i]);

			if (vkEndCommandBuffer(_command_buffers[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to record command buffer");

			
		}
	}

	void create_semaphores()
	{
		VkSemaphoreCreateInfo semaphore_info = {};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		if (vkCreateSemaphore(_device, &semaphore_info, nullptr, &_image_available_semaphore) != VK_SUCCESS ||
			vkCreateSemaphore(_device, &semaphore_info, nullptr, &_render_finished_semaphore) != VK_SUCCESS)
			throw std::runtime_error("Failed to create semaphores");
	}

	VkShaderModule create_shader_module(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shader_module;
		if (vkCreateShaderModule(_device, &create_info, nullptr, &shader_module) != VK_SUCCESS)
			throw std::runtime_error("Failed to create shader module");

		return shader_module;
	}

	void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
	{
		VkCommandBufferAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandPool = _command_pool;
		alloc_info.commandBufferCount = 1;

		VkCommandBuffer command_buffer;
		vkAllocateCommandBuffers(_device, &alloc_info, &command_buffer);

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(command_buffer, &begin_info);

		VkBufferCopy copy_region = {};
		copy_region.srcOffset = 0;
		copy_region.dstOffset = 0;
		copy_region.size = size;

		vkCmdCopyBuffer(command_buffer, src, dst, 1, &copy_region);

		vkEndCommandBuffer(command_buffer);

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;

		vkQueueSubmit(_gfx_queue, 1, &submit_info, VK_NULL_HANDLE);
		vkQueueWaitIdle(_gfx_queue);

		vkFreeCommandBuffers(_device, _command_pool, 1, &command_buffer);
	}

	bool is_device_suitable(VkPhysicalDevice device)
	{
		QueueFamilyIndices i = find_queue_families(device);

		bool extension_supported = check_device_extension_support(device);
		bool swap_chain_adequate = false;

		if (extension_supported)
		{
			SwapChainSupportDetails swap_chain_support = query_swap_chain_support(device);
			swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
		}

		if (i.is_complete())
			_queue_family_indices = i;

		return i.is_complete() && extension_supported && swap_chain_adequate;
	}

	uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(_physical_device, &mem_properties);

		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
			if (type_filter & (1 << i) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
				return i;

		throw std::runtime_error("Failed to find suitable memory type");
	}

	bool check_device_extension_support(VkPhysicalDevice device)
	{
		uint32_t extension_count;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

		std::vector<VkExtensionProperties> available_extensions(extension_count);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

		std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

		for (const auto& extension : available_extensions)
			required_extensions.erase(extension.extensionName);

		return required_extensions.empty();
	}

	void setup_debug_callback(void)
	{
		if (!enable_validation_layers)
			return;

		VkDebugReportCallbackCreateInfoEXT create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
		create_info.pfnCallback = debug_callback;

		if (CreateDebugReportCallbackEXT(_instance, &create_info, nullptr, &_callback) != VK_SUCCESS)
			throw std::runtime_error("Failed to set up debug callback");
	}

	bool check_validation_layer_support(void)
	{
		uint32_t layer_count;
		vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

		std::vector<VkLayerProperties> available_layers(layer_count);
		vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

		for (const char* layer_name : validation_layers)
		{
			bool layer_found = false;

			for (const auto& layer_properties : available_layers)
			{
				if (strcmp(layer_name, layer_properties.layerName) == 0)
				{
					layer_found = true;
					break;
				}
			}
			if (!layer_found)
				return false;
		}

		return true;
	}

	VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats)
	{
		if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED)
			return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };

		for (const auto& available_format : available_formats)
			if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return available_format;

		return available_formats[0];
	}
	
	VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR> available_present_modes)
	{
		VkPresentModeKHR best = VK_PRESENT_MODE_FIFO_KHR;

		for (const auto& available_present_mode : available_present_modes)
			if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
				return available_present_mode;
			else if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
				best = available_present_mode;

		return best;
	}

	VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else
		{
			VkExtent2D actual = { WIDTH, HEIGHT };

			actual.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual.width));
			actual.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual.height));

			return actual;
		}
	}

	std::vector<const char*> get_required_extensions(void)
	{
		uint32_t glfw_extension_count = 0;
		const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

		std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

		if (enable_validation_layers)
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

		return extensions;
	}

	QueueFamilyIndices find_queue_families(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

		std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

		int i = 0;
		for (const auto& queue_family : queue_families)
		{
			if (queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphics_family = i;

			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &present_support);

			if (queue_family_count > 0 && present_support)
				indices.present_family = i;

			if (indices.is_complete())
				break;

			i++;
		}

		return indices;
	}

	SwapChainSupportDetails query_swap_chain_support(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _surface, &details.capabilities);

		uint32_t format_count;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count, nullptr);

		if (format_count > 0)
		{
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count, details.formats.data());
		}

		uint32_t present_mode_count;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &present_mode_count, nullptr);

		if (present_mode_count > 0)
		{
			details.present_modes.resize(present_mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &present_mode_count, details.present_modes.data());
		}

		return details;
	}

	static void on_window_resize(GLFWwindow* window, int width, int height)
	{
		hello_triangle_application* app = reinterpret_cast<hello_triangle_application*>(glfwGetWindowUserPointer(window));
		app->recreate_swapchain();
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT obj_type, uint64_t obj, size_t location, int32_t code, const char* layer_prefix, const char* msg, void* user_data)
	{
		std::cerr << "Validation layer: " << msg << std::endl;

		return VK_FALSE;
	}
};

int main(void)
{
	hello_triangle_application app;

	try
	{
		app.run();
	}
	catch(const std::runtime_error& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}