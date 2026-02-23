type t =
  {
    value : float; (* scalar value of this node calculated during forward pass *)
    mutable grad : float; (* derivative of the loss w.r.t. this node, calculated in backward pass *)
    children : t list; (* children of this node in the computation graph *)
    local_grads : float list; (* local derivative of this node w.r.t. its children *)
  }

let value a = a.value

let grad a = a.grad

let make value children local_grads =
  { value; grad = 0.; children; local_grads}

let const value =
  make value [] []

let add a b =
  make (value a +. value b) [a; b] [1.; 1.]

let mul a b =
  make (value a *. value b) [a; b] [value b; value a]

let backward a =
  (* Breadth-first search (see the example from the Queue module). *)
  let topo a =
    let queue = Queue.create() in
    let ans = ref [] in
    Queue.push a queue;
    let rec loop () =
      if Queue.is_empty queue then !ans
      else explore @@ Queue.pop queue
    and explore a =
      (* TODO: could be faster, e.g. by having a visited field in nodes. *)
      if not (List.mem a !ans) then (
        ans := a :: !ans;
        List.iter (fun a -> Queue.push a queue) a.children;
        loop ()
      ) else loop ()
    in
    loop()
  in
  let topo = List.rev @@ topo a in
  a.grad <- 1.;
  List.iter (fun a -> List.iter2 (fun child grad -> child.grad <- child.grad +. grad) a.children a.local_grads) topo

module Inline = struct
  let ( + ) = add
  let ( * ) = mul
end

(* Basic test. *)
let () =
  let open Inline in
  let a = const 2. in
  let b = const 3. in
  let c = a * b in
  let l = c + a in
  backward l;
  assert (grad a = 4.);
  assert (grad b = 2.)
