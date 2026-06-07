package com.example.salesvoice.ui.products

import android.app.AlertDialog
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.lifecycleScope
import com.example.salesvoice.R
import com.example.salesvoice.data.model.Product
import com.example.salesvoice.databinding.DialogAddProductBinding
import com.example.salesvoice.databinding.FragmentProductsBinding
import com.example.salesvoice.ui.ViewModelFactory
import kotlinx.coroutines.launch

class ProductsFragment : Fragment() {

    private var _binding: FragmentProductsBinding? = null
    private val binding get() = _binding!!

    private val viewModel: ProductsViewModel by viewModels {
        ViewModelFactory(requireContext())
    }

    private lateinit var adapter: ProductAdapter
    private var currencySymbol = "₹"

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentProductsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Load settings
        val sharedPrefs = requireActivity().getSharedPreferences("SalesVoicePrefs", Context.MODE_PRIVATE)
        currencySymbol = sharedPrefs.getString("pref_currency", "₹") ?: "₹"

        setupRecyclerView()
        observeProducts()

        binding.fabAddProduct.setOnClickListener {
            showAddEditDialog(null)
        }
    }

    private fun setupRecyclerView() {
        adapter = ProductAdapter(
            currencySymbol = currencySymbol,
            onEditClick = { product -> showAddEditDialog(product) },
            onDeleteClick = { product -> showDeleteConfirmation(product) }
        )
        binding.rvProducts.adapter = adapter
    }

    private fun observeProducts() {
        viewModel.allProducts.observe(viewLifecycleOwner) { products ->
            if (products.isNullOrEmpty()) {
                binding.llEmptyState.visibility = View.VISIBLE
                binding.rvProducts.visibility = View.GONE
            } else {
                binding.llEmptyState.visibility = View.GONE
                binding.rvProducts.visibility = View.VISIBLE
                adapter.submitList(products)
            }
        }
    }

    private fun showAddEditDialog(product: Product?) {
        val dialogBinding = DialogAddProductBinding.inflate(layoutInflater)
        val builder = AlertDialog.Builder(requireContext(), com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
        builder.setView(dialogBinding.root)
        val alertDialog = builder.create()

        // Configure Prefix text with currency symbol
        dialogBinding.tilPrice.prefixText = currencySymbol
        dialogBinding.tilProfit.prefixText = currencySymbol

        // Setup Spinner
        val units = arrayOf("kg", "g", "litre", "piece", "dozen", "packet")
        val spinnerAdapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_item, units)
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        dialogBinding.spinnerUnit.adapter = spinnerAdapter

        // Pre-fill fields if editing
        val isEditing = product != null
        if (isEditing && product != null) {
            dialogBinding.tvDialogTitle.text = getString(R.string.products_edit)
            dialogBinding.etName.setText(product.name)
            dialogBinding.etAliases.setText(product.aliases)
            dialogBinding.etPrice.setText(product.pricePerUnit.toString())
            dialogBinding.etProfit.setText(product.profitPerUnit.toString())
            val unitIndex = units.indexOf(product.unit)
            if (unitIndex >= 0) {
                dialogBinding.spinnerUnit.setSelection(unitIndex)
            }
        } else {
            dialogBinding.tvDialogTitle.text = getString(R.string.products_add)
        }

        dialogBinding.btnCancel.setOnClickListener {
            alertDialog.dismiss()
        }

        dialogBinding.btnSave.setOnClickListener {
            val name = dialogBinding.etName.text.toString().trim()
            val aliases = dialogBinding.etAliases.text.toString().trim()
            val selectedUnit = dialogBinding.spinnerUnit.selectedItem.toString()
            val priceStr = dialogBinding.etPrice.text.toString().trim()
            val profitStr = dialogBinding.etProfit.text.toString().trim()

            // Validate Name
            if (name.isEmpty()) {
                dialogBinding.tilName.error = getString(R.string.products_name_empty_error)
                return@setOnClickListener
            } else {
                dialogBinding.tilName.error = null
            }

            // Validate Price
            val price = priceStr.toDoubleOrNull()
            if (price == null || price <= 0) {
                dialogBinding.tilPrice.error = getString(R.string.products_price_error)
                return@setOnClickListener
            } else {
                dialogBinding.tilPrice.error = null
            }

            // Validate Profit
            val profit = profitStr.toDoubleOrNull()
            if (profit == null || profit < 0 || profit >= price) {
                dialogBinding.tilProfit.error = getString(R.string.products_profit_error)
                return@setOnClickListener
            } else {
                dialogBinding.tilProfit.error = null
            }

            // Check Duplicate Name in Room
            lifecycleScope.launch {
                val isDuplicate = viewModel.isDuplicateName(name, product?.id)
                if (isDuplicate) {
                    dialogBinding.tilName.error = getString(R.string.products_duplicate_error)
                } else {
                    dialogBinding.tilName.error = null
                    
                    if (isEditing && product != null) {
                        val updatedProduct = product.copy(
                            name = name,
                            aliases = aliases,
                            unit = selectedUnit,
                            pricePerUnit = price,
                            profitPerUnit = profit
                        )
                        viewModel.updateProduct(updatedProduct)
                        Toast.makeText(context, getString(R.string.products_updated), Toast.LENGTH_SHORT).show()
                    } else {
                        val newProduct = Product(
                            name = name,
                            aliases = aliases,
                            unit = selectedUnit,
                            pricePerUnit = price,
                            profitPerUnit = profit
                        )
                        viewModel.insertProduct(newProduct) {
                            // Long ID callback
                        }
                        Toast.makeText(context, getString(R.string.products_added), Toast.LENGTH_SHORT).show()
                    }
                    alertDialog.dismiss()
                }
            }
        }

        alertDialog.show()
    }

    private fun showDeleteConfirmation(product: Product) {
        AlertDialog.Builder(requireContext(), com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
            .setTitle(getString(R.string.products_delete_title))
            .setMessage(getString(R.string.products_delete_message, product.name))
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                viewModel.deleteProduct(product)
                Toast.makeText(context, getString(R.string.products_deleted, product.name), Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
